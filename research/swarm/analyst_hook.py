"""The analyst's PreToolUse hook (DRIVER_PLAN D2): an analyst runs the driver's commands and
nothing else.

Claude Code runs this script before each tool call, with the call as JSON on standard input
(``tool_name``, ``tool_input``, ``cwd``, and ``agent_type`` when a subagent makes the call). For an
``analyst`` it denies, with a plain reason the analyst reads:

* every tool but Bash (the agent type's own ``tools: Bash`` and ``disallowedTools`` list say the
  same; this is the second lock);
* every Bash command that is not exactly the driver's invocation: ``python``, ``python3`` or
  ``scripts/py``, then ``-m research.swarm.driver`` and one of the analyst's commands
  (:data:`ANALYST_COMMANDS`; ``start`` and ``stop`` are the orchestrator's), from the repo root,
  with none of ``;``, ``&``, ``|``, a backtick, ``$``, ``<``, ``>`` or a newline anywhere in it, so
  nothing is chained, substituted, expanded or redirected.

For every other agent type, and for the session itself, it says nothing: the call goes on to the
permissions as it would without the hook. An allowed analyst command is not approved here either;
the permissions decide it. Input that does not read as a hook call is refused (exit 2), since
whether the caller is an analyst cannot be told.

Wired only in ``.claude/settings.local.json`` on the owner's machine, never committed:

    {"hooks": {"PreToolUse": [{"matcher": "*", "hooks": [{"type": "command",
        "command": "python3 <repo>/research/swarm/analyst_hook.py"}]}]}}

The script uses the standard library only, so any Python 3 runs it.
"""
from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
ANALYST = "analyst"
INTERPRETERS = ("python", "python3", "scripts/py", "./scripts/py")
MODULE = "research.swarm.driver"
# The driver's commands an analyst runs (``research.swarm.driver.ANALYST_COMMANDS``; a test holds
# the two lists equal).
ANALYST_COMMANDS = ("ask", "preview", "answer", "revisit", "fit", "results", "seal", "export")
# Anything that chains, substitutes, expands or redirects in a shell.
FORBIDDEN = (";", "&", "|", "`", "$", "<", ">", "\n", "\r")
FORM = ("scripts/py -m research.swarm.driver <command> --run <id> ..., run from the repo root, "
        "where <command> is one of " + ", ".join(ANALYST_COMMANDS))


def deny(reason: str) -> dict[str, Any]:
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                   "permissionDecisionReason": reason}}


def bash_refusal(command: Any, cwd: Any) -> str | None:
    """Why an analyst's Bash ``command`` is not the driver's invocation, or None when it is."""
    if not isinstance(command, str) or not command.strip():
        return f"An analyst runs only the driver's commands: {FORM}."
    found = [c for c in FORBIDDEN if c in command]
    if found:
        shown = ", ".join("a line break" if c == "\n" else "a backtick" if c == "`" else f"'{c}'"
                          for c in dict.fromkeys("\n" if c == "\r" else c for c in found))
        return (f"The command holds {shown}, which an analyst's command never holds, in a reason "
                f"either: one driver command at a time, nothing chained, substituted or "
                f"redirected. The form is {FORM}.")
    try:
        words = shlex.split(command)
    except ValueError:
        return f"The command does not read as one driver command (an unclosed quote?). The form is {FORM}."
    if len(words) < 4 or words[0] not in INTERPRETERS or words[1] != "-m" or words[2] != MODULE:
        return f"An analyst runs only the driver's commands: {FORM}."
    if words[3] not in ANALYST_COMMANDS:
        return (f"{words[3]!r} is not an analyst's command; the analyst's commands are "
                f"{', '.join(ANALYST_COMMANDS)}.")
    try:
        here = Path(str(cwd)).resolve() if cwd else None
    except (OSError, RuntimeError):
        here = None
    if here != REPO:
        return f"The driver runs from the repo root, {REPO}, not from {cwd or 'an unknown folder'}."
    return None


def decide(call: Any) -> dict[str, Any] | None:
    """The hook's answer to one tool call: a denial, or None to say nothing."""
    if not isinstance(call, dict) or call.get("agent_type") != ANALYST:
        return None
    tool = call.get("tool_name")
    if tool != "Bash":
        return deny(f"An analyst works only through the driver's commands, run with Bash; {tool} "
                    f"is not open to it.")
    why = bash_refusal((call.get("tool_input") or {}).get("command"), call.get("cwd"))
    return deny(why) if why else None


def main(stdin: Any = None, stdout: Any = None, stderr: Any = None) -> int:
    stdin, stdout, stderr = stdin or sys.stdin, stdout or sys.stdout, stderr or sys.stderr
    try:
        call = json.loads(stdin.read() or "")
    except ValueError:
        stderr.write("The analyst hook could not read the tool call, so it is refused.\n")
        return 2
    answer = decide(call)
    if answer is not None:
        stdout.write(json.dumps(answer) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
