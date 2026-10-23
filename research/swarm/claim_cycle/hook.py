"""The claim cycle's PreToolUse hook: a theorist or an experimentalist runs the claim cycle's tools
and nothing else, so it reads a record only through the view (``view.py``) and never the sealed
file by any other tool.

Claude Code runs this script before each tool call, with the call as JSON on standard input
(``tool_name``, ``tool_input``, ``cwd``, and ``agent_type`` when a subagent makes the call). For a
``theorist`` or an ``experimentalist`` it approves, with Claude Code's PreToolUse allow
(``analyst_hook.allow``), the harness's report tools (``analyst_hook.REPORT_TOOLS``), by which it
hands its report back, whether or not a cycle is pinned, and a tools command it has checked whole
under the pin. It denies, with a plain reason the agent reads:

* every other tool;
* every Bash command that is not exactly the tools' invocation: ``scripts/py``, then
  ``-m research.swarm.claim_cycle.tools`` and one of its role's commands (:data:`COMMANDS`: the
  experimentalist proposes nothing), from the repo root, that no shell expands or runs anything
  in, under the analyst hook's rules (``analyst_hook.shell_refusal``: none of ``; & | $ < >``, a
  backtick or a line break anywhere, quoted or not; outside quotes only plain characters; so a
  proposal's JSON goes in single quotes), whose refusals name the theorist or the experimentalist
  and a tools command (:data:`COMMAND_NAME`), never an analyst or the driver. ``python``,
  ``python3`` and ``./scripts/py``, which it passed before 2026-10-23, and an environment setting
  before the command, such as the prefix the root CLAUDE.md gives builders, are each refused in
  words that name the one form (``analyst_hook.start_refusal``). The repo root is the checkout
  this file lies in (:data:`REPO`, read from the file's own path), named by an absolute path:
  another checkout's root, a worktree's say, holds its own ``scripts/py`` and tools, which may
  differ from this one's;
* every Bash call with a truthy ``run_in_background`` (``analyst_hook.background_refusal``): an
  agent of the claim cycle runs one tools command at a time, in the foreground; and every one with
  a truthy ``dangerouslyDisableSandbox`` (``analyst_hook.sandbox_refusal``): a tools command it
  approves runs under whatever sandbox the session has, never outside it;
* every tools command while no cycle is pinned, and every one that names a folder or a file other
  than the pinned cycle's corpus folder and proposals file (``pin.py``: one cycle at a time, each
  with a proposals file of its own, so no agent reads both the corpus and its shuffled copy, nor
  the other cycle's proposals). The command is read with the tools' own
  parser (``pin.build_parser``), so what the hook checks is what the tools read.

For every other caller it says nothing, neither approving nor denying: the call goes on to the
permissions. Input that does not read as a hook call is refused (exit 2).

**Wiring**, only in ``.claude/settings.local.json`` on the owner's machine, beside the analyst's
hook, never committed. The settings hold no allow rule for the tools: this hook approves a
theorist's or an experimentalist's checked calls itself. A prefix rule such as
``Bash(scripts/py -m research.swarm.claim_cycle.tools:*)`` is not wanted (the audit of
2026-10-23): it approves those words for every caller, from whatever folder the caller is in, so
it runs whatever ``scripts/py`` and tools that folder holds, which any agent that can write files
can change, and it approves another caller's tools command on any folder, the pin unread. The
hook is one PreToolUse entry for
every tool (``"matcher": "*"``), read from the project's own checkout, the folder the claim
cycle's agents run from and the one whose pin the orchestrator writes, and its answer reaches
Claude Code unchanged:

    {"hooks": {"PreToolUse": [{"matcher": "*", "hooks": [{"type": "command",
        "command": "python3 \\"$CLAUDE_PROJECT_DIR/research/swarm/claim_cycle/hook.py\\""}]}]}}

The owner's entry wraps that command so that it fails closed: a theorist's or an
experimentalist's call is refused (exit 2) when the hook's file is missing, unless it is a report
tool, and when the hook exits with a code other than 0 or 2; it refuses no other caller's call.

It uses the standard library, the analyst hook and the pin, which use only the standard library, so
any Python 3 runs it.
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


def _beside(name: str, path: Path) -> Any:
    """The module ``name`` at ``path``, loaded from its file (run as a script, the repo is not on
    the path)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


def _analyst_hook() -> Any:
    try:
        from research.swarm import analyst_hook
    except ImportError:  # run as a script, with the repo not on the path
        analyst_hook = _beside("analyst_hook", HERE.parent / "analyst_hook.py")
    return analyst_hook


def _pin() -> Any:
    try:
        from research.swarm.claim_cycle import pin
    except ImportError:  # run as a script, with the repo not on the path
        pin = _beside("claim_cycle_pin", HERE / "pin.py")
    return pin


A = _analyst_hook()
N = _pin()
# The pin the hook reads: ``pin.PIN``'s place (``.claude/claim_cycle.json``) under the hook's own
# repo root, whichever checkout the pin module was imported from.
PIN = REPO / N.PIN.relative_to(N.REPO)
THEORIST = "theorist"
EXPERIMENTALIST = "experimentalist"
MODULE = "research.swarm.claim_cycle.tools"
# What the refusals the analyst hook shares with this one call the command.
COMMAND_NAME = "tools command"
# Each role's commands (``tools.COMMANDS``; a test holds the two lists equal).
COMMANDS = {THEORIST: ("index", "records", "proposals", "propose"),
            EXPERIMENTALIST: ("index", "records", "proposals")}
REPORT_TOOLS = A.REPORT_TOOLS
FORBIDDEN = A.FORBIDDEN
deny = A.deny
allow = A.allow


def form(role: str) -> str:
    return (f"{A.INTERPRETER} -m {MODULE} <command> ..., run from the repo root, where <command> "
            f"is one of {', '.join(COMMANDS[role])}")


def named(role: str) -> str:
    """``role`` with its article: "a theorist", "an experimentalist"."""
    return f"{'an' if role[:1] in 'aeiou' else 'a'} {role}"


def bash_refusal(command: Any, cwd: Any, role: str) -> str | None:
    """Why ``role``'s Bash ``command`` is not the tools' invocation under the pin, or None when it
    is."""
    if not isinstance(command, str) or not command.strip():
        return f"{named(role).capitalize()} runs only the claim cycle's tools: {form(role)}."
    why = A.shell_refusal(command, subject=named(role), name=COMMAND_NAME, form=form(role),
                          quoted=None)
    if why:
        return why
    try:
        words = shlex.split(command)
    except ValueError:
        return f"The command does not read as one tools command (an unclosed quote?): {form(role)}."
    started = A.start_refusal(words[0], "The tools run", form(role)) if words else None
    if started:
        return started
    if len(words) < 4 or words[0] != A.INTERPRETER or words[1] != "-m" or words[2] != MODULE:
        return f"{named(role).capitalize()} runs only the claim cycle's tools: {form(role)}."
    if words[3] not in COMMANDS[role]:
        return (f"{words[3]!r} is not a command {named(role)} runs; its commands are "
                f"{', '.join(COMMANDS[role])}.")
    if not A.at_root(cwd, REPO):
        return f"The tools run from the repo root, {REPO}, not from {cwd or 'an unknown folder'}."
    return N.refusal(words[3:], role, pin=PIN, repo=REPO)


def decide(call: Any) -> dict[str, Any] | None:
    """The hook's answer to one tool call: for a theorist or an experimentalist, an approval of a
    report tool or of a tools command it checked whole under the pin (:func:`bash_refusal`) and
    runs in the foreground and the sandbox (``analyst_hook.background_refusal``,
    ``analyst_hook.sandbox_refusal``), and a denial of anything else; for every other caller,
    None, to say nothing."""
    if not isinstance(call, dict) or call.get("agent_type") not in COMMANDS:
        return None
    role = call["agent_type"]
    tool = call.get("tool_name")
    if tool in REPORT_TOOLS:
        return allow(f"{named(role).capitalize()} hands its report back with {tool}.")
    if tool != "Bash":
        return deny(f"{named(role).capitalize()} works only through the claim cycle's tools, run "
                    f"with Bash; {tool} is not open to it.")
    given = call.get("tool_input")
    command = given.get("command") if isinstance(given, dict) else None
    why = (bash_refusal(command, call.get("cwd"), role)
           or A.background_refusal(given, named(role), COMMAND_NAME)
           or A.sandbox_refusal(given, named(role), COMMAND_NAME))
    if why:
        return deny(why)
    return allow(f"The claim cycle's hook checked this tools command ({shlex.split(command)[3]}) "
                 f"whole: the one form, run from the repo root, naming only the pinned cycle's "
                 f"corpus folder or proposals file, with nothing a shell expands or runs.")


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
