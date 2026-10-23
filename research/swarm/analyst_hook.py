"""The analyst's PreToolUse hook (DRIVER_PLAN D2): an analyst runs the driver's commands and
nothing else.

Claude Code runs this script before each tool call, with the call as JSON on standard input
(``tool_name``, ``tool_input``, ``cwd``, and ``agent_type`` when a subagent makes the call). For an
``analyst`` it approves, with Claude Code's PreToolUse allow (:func:`allow`), the harness's two
report tools, ``SubagentHandback`` and ``StructuredOutput`` (:data:`REPORT_TOOLS`), by which an
analyst hands its report back (the live check of 2026-10-22 found them denied here, so no
analyst's report reached the orchestrator; a report says what the analyst did, and the run's
record is never read from it: the driver's ``account`` is the analyst's account), and a Bash
command it has checked whole: the driver's invocation, ``scripts/py -m research.swarm.driver``
and one of the analyst's commands (:data:`ANALYST_COMMANDS`; ``start`` and ``stop`` are the
orchestrator's), from the repo root, that no shell expands or runs anything in. It denies, with a
plain reason the analyst reads, every other tool (the agent type's own ``tools: Bash`` and
``disallowedTools`` list say the same; this is the second lock) and every other command:

* one that starts otherwise than ``scripts/py`` (:data:`INTERPRETER`): ``python``, ``python3``
  and ``./scripts/py``, which it passed before 2026-10-23, and an environment setting before the
  command, such as the prefix the root CLAUDE.md gives builders, are each refused in words that
  name the one form, since the hook approves only what it checked, and ``python`` is whatever the
  caller's PATH finds first;
* one run from anywhere but the repo root, the checkout this file lies in (:data:`REPO`, read from
  the file's own path), named by an absolute path: another checkout's root, a worktree's say,
  holds its own ``scripts/py`` and driver, which may differ from this one's;
* one that a shell could read as more than one program's words:

  - none of ``;``, ``&``, ``|``, a backtick, ``$``, ``<``, ``>``, a line break or an ASCII control
    character other than a tab anywhere in it (:data:`FORBIDDEN`, then each code point below 32,
    and 127), quoted or not, so nothing is chained, substituted or redirected (a shell reads a tab
    between words as a space, and in quotes as itself);
  - outside quotes, only the characters of :data:`PLAIN` and spaces, and ``=`` only inside a word
    after one of them, so no glob, tilde, brace, history, comment, subshell, ``=(...)`` or
    ``=cmd`` of bash's or zsh's, whatever their options, is ever read (zsh is the shell an
    analyst's Bash runs on the owner's machine: its glob qualifiers ``*(e:...:)`` and process
    substitution ``=(...)`` run programs);
  - inside double quotes, no backslash and no ``!``; inside single quotes, anything else;

* one run in the background, a Bash call with a truthy ``run_in_background``, whatever its
  command: an analyst runs one driver command at a time, in the foreground, so that it reads what
  each prints before it runs the next;
* one let out of the sandbox, a Bash call with a truthy ``dangerouslyDisableSandbox``, whatever
  its command: a command the hook approves runs under whatever sandbox the session has, never
  outside it;
* a Bash call whose input is not an object, which holds no command to check.

Its refusals of the shell's rules (:func:`shell_refusal`), of the background
(:func:`background_refusal`) and of leaving the sandbox (:func:`sandbox_refusal`) name the agent
and its command, here an analyst and a driver command (:data:`SUBJECT`, :data:`COMMAND_NAME`), so
that the claim cycle's hook, which shares them, names its own.

For every other agent type, and for the session itself, it says nothing, neither approving nor
denying: the call goes on to the permissions as it would without the hook. Input that does not
read as a hook call is refused (exit 2), since whether the caller is an analyst cannot be told.

**Wiring**, only in ``.claude/settings.local.json`` on the owner's machine, never committed. The
settings hold no allow rule for the driver: this hook approves an analyst's checked calls itself.
A prefix rule such as ``Bash(scripts/py -m research.swarm.driver:*)`` is not wanted (the audit
of 2026-10-23): it approves those words for every caller, from whatever folder the caller is in,
so it runs whatever ``scripts/py`` and driver that folder holds, which any agent that can write
files can change. The hook is one PreToolUse entry for every tool (``"matcher": "*"``; a run's
``start`` refuses without one, ``driver.hook_wired``), read from the project's own checkout, the
folder the analysts run from, so that its root is theirs, and its answer reaches Claude Code
unchanged:

    {"hooks": {"PreToolUse": [{"matcher": "*", "hooks": [{"type": "command",
        "command": "python3 \\"$CLAUDE_PROJECT_DIR/research/swarm/analyst_hook.py\\""}]}]}}

The owner's entry wraps that command so that it fails closed: an analyst's call is refused
(exit 2) when the hook's file is missing, unless it is a report tool, and when the hook exits with
a code other than 0 or 2; it refuses no other caller's call.

The script uses the standard library only, so any Python 3 runs it.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import string
import sys
from pathlib import Path
from typing import Any

# The checkout this file lies in: the one root it approves a command from.
REPO = Path(__file__).resolve().parents[2]
ANALYST = "analyst"
# How an analyst's refusals name it and its command, and what it puts in quotes.
SUBJECT = "an analyst"
COMMAND_NAME = "driver command"
QUOTED = "a reason"
# The one way a command it approves starts: the repo's own launcher, read from the repo root.
INTERPRETER = "scripts/py"
# The ways of starting Python it passed before 2026-10-23, each refused now in words that name the
# one form.
OTHER_INTERPRETERS = ("python", "python3", "./scripts/py")
# A first word a shell reads as an environment setting for the command after it.
SETTING = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
MODULE = "research.swarm.driver"
# The driver's commands an analyst runs (``research.swarm.driver.ANALYST_COMMANDS``; a test holds
# the two lists equal).
ANALYST_COMMANDS = ("ask", "preview", "answer", "revisit", "fit", "results", "seal", "export",
                    "want", "account")
# The harness's tools by which a subagent hands its report back, exactly as named.
REPORT_TOOLS = ("SubagentHandback", "StructuredOutput")
# Anything that chains, substitutes or redirects in a shell, quoted or not.
FORBIDDEN = (";", "&", "|", "`", "$", "<", ">", "\n", "\r")
# What a word holds outside quotes: no shell reads any of these as anything but itself.
PLAIN = frozenset(string.ascii_letters + string.digits + "._-/:,@%+")
SHOWN_PLAIN = "letters, digits and . _ - / : , @ % +"
BLANKS = (" ", "\t")
# Inside double quotes a shell still reads these (an escape; zsh's and bash's history).
ACTIVE_IN_DOUBLE = ("\\", "!")
FORM = ("scripts/py -m research.swarm.driver <command> --run <id> ..., run from the repo root, "
        "where <command> is one of " + ", ".join(ANALYST_COMMANDS))


def quote_it(quoted: str | None) -> str:
    """What a word holds outside quotes, and where anything else goes; ``quoted`` names what the
    agent puts in quotes ("a reason"), or is None where it names nothing."""
    such_as = f", such as {quoted}," if quoted else ""
    return (f"Outside quotes a word holds only {SHOWN_PLAIN}, and = inside a word; put anything "
            f"else{such_as} in double or single quotes.")


def deny(reason: str) -> dict[str, Any]:
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                   "permissionDecisionReason": reason}}


def allow(reason: str) -> dict[str, Any]:
    """Claude Code's PreToolUse approval: the call runs without the permissions' prompt."""
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow",
                                   "permissionDecisionReason": reason}}


def at_root(cwd: Any, root: Path) -> bool:
    """Whether ``cwd``, the folder a call runs in, is ``root``: an absolute path that resolves to
    it. A relative one never is, since it would be read from wherever the hook runs."""
    if not isinstance(cwd, (str, os.PathLike)):
        return False
    try:
        path = Path(cwd)
        return path.is_absolute() and path.resolve() == root
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def start_refusal(first: str, runs: str, form: str) -> str | None:
    """Why a command whose first word is ``first`` is refused for how it starts, in words that
    name ``form``, the one form (``runs`` says what runs: "The driver runs"); None where ``first``
    is neither an environment setting nor one of :data:`OTHER_INTERPRETERS`."""
    if SETTING.match(first):
        return (f"The command starts with an environment setting, {first}, which the hook "
                f"refuses: give the command alone, with nothing before it. The one form is {form}.")
    if first in OTHER_INTERPRETERS:
        return f"{runs} only through {INTERPRETER}, not {first}. The one form is {form}."
    return None


def _shown(c: str) -> str:
    return {"\n": "a line break", "\r": "a line break", "`": "a backtick", "\t": "a tab",
            "\\": "a backslash"}.get(c) or (f"'{c}'" if c.isprintable() else f"the character U+{ord(c):04X}")


def shell_refusal(command: str, *, subject: str, name: str, form: str,
                  quoted: str | None) -> str | None:
    """Why a shell could read ``command`` as more than one program's words, or None: the rules in
    the module docstring, checked character by character with the quotes followed as bash and
    zsh follow them. The words name the agent, ``subject`` with its article ("an analyst"), its
    command's ``name`` ("driver command") and ``form``, and what it puts in quotes, ``quoted``
    ("a reason"; None for nothing), so each hook's refusals speak of its own."""
    found = [c for c in FORBIDDEN if c in command]
    if found:
        shown = ", ".join(dict.fromkeys(_shown(c) for c in found))
        return (f"The command holds {shown}, which {subject}'s command never holds, in "
                f"{quoted or 'quotes'} either: one {name} at a time, nothing chained, substituted "
                f"or redirected. The form is {form}.")
    control = next((c for c in command if (ord(c) < 32 or ord(c) == 127) and c != "\t"), None)
    if control is not None:
        return f"The command holds {_shown(control)}, a control character. The form is {form}."
    quote: str | None = None
    after_plain = False  # the last character read was a plain one outside quotes, in this word
    for c in command:
        if quote is not None:
            if c == quote:
                quote = None
            elif quote == '"' and c in ACTIVE_IN_DOUBLE:
                return (f"The command holds {_shown(c)} inside double quotes, which a shell still "
                        f"reads there; leave it out, or put the words in single quotes.")
            after_plain = False
            continue
        if c in BLANKS or c in "'\"":
            quote = c if c in "'\"" else None
            after_plain = False
        elif c in PLAIN:
            after_plain = True
        elif c == "=" and after_plain:
            after_plain = False
        else:
            return (f"The command holds {_shown(c)} outside quotes, which a shell can expand or "
                    f"run. {quote_it(quoted)}")
    if quote is not None:
        return f"The command does not read as one {name} (an unclosed quote?). The form is {form}."
    return None


def background_refusal(given: Any, subject: str, name: str) -> str | None:
    """Why a Bash call whose input is ``given`` is refused for where it runs: a truthy
    ``run_in_background``, in words that name ``subject`` and its command's ``name``, as
    :func:`shell_refusal`'s do; None for a call run in the foreground."""
    if isinstance(given, dict) and given.get("run_in_background"):
        return (f"{subject.capitalize()} runs one {name} at a time, in the foreground: run it "
                f"again without run_in_background.")
    return None


def sandbox_refusal(given: Any, subject: str, name: str) -> str | None:
    """Why a Bash call whose input is ``given`` is refused for asking to leave the sandbox: a
    truthy ``dangerouslyDisableSandbox``, in words that name ``subject`` and its command's
    ``name``, as :func:`background_refusal`'s do; None for a call that does not ask."""
    if isinstance(given, dict) and given.get("dangerouslyDisableSandbox"):
        return (f"{subject.capitalize()} never runs a {name} outside the sandbox: run it again "
                f"without dangerouslyDisableSandbox.")
    return None


def bash_refusal(command: Any, cwd: Any) -> str | None:
    """Why an analyst's Bash ``command`` is not the driver's invocation, or None when it is."""
    if not isinstance(command, str) or not command.strip():
        return f"An analyst runs only the driver's commands: {FORM}."
    why = shell_refusal(command, subject=SUBJECT, name=COMMAND_NAME, form=FORM, quoted=QUOTED)
    if why:
        return why
    try:
        words = shlex.split(command)
    except ValueError:
        return f"The command does not read as one driver command (an unclosed quote?). The form is {FORM}."
    started = start_refusal(words[0], "The driver runs", FORM) if words else None
    if started:
        return started
    if len(words) < 4 or words[0] != INTERPRETER or words[1] != "-m" or words[2] != MODULE:
        return f"An analyst runs only the driver's commands: {FORM}."
    if words[3] not in ANALYST_COMMANDS:
        return (f"{words[3]!r} is not an analyst's command; the analyst's commands are "
                f"{', '.join(ANALYST_COMMANDS)}.")
    if not at_root(cwd, REPO):
        return f"The driver runs from the repo root, {REPO}, not from {cwd or 'an unknown folder'}."
    return None


def decide(call: Any) -> dict[str, Any] | None:
    """The hook's answer to one tool call: for an analyst, an approval of a report tool or of a
    driver command it checked whole (:func:`bash_refusal`) and runs in the foreground and the
    sandbox (:func:`background_refusal`, :func:`sandbox_refusal`), and a denial of anything else,
    a call whose input is not an object among them; for every other caller, None, to say
    nothing."""
    if not isinstance(call, dict) or call.get("agent_type") != ANALYST:
        return None
    tool = call.get("tool_name")
    if tool in REPORT_TOOLS:
        return allow(f"An analyst hands its report back with {tool}.")
    if tool != "Bash":
        return deny(f"An analyst works only through the driver's commands, run with Bash; {tool} "
                    f"is not open to it.")
    given = call.get("tool_input")
    command = given.get("command") if isinstance(given, dict) else None
    why = (bash_refusal(command, call.get("cwd"))
           or background_refusal(given, SUBJECT, COMMAND_NAME)
           or sandbox_refusal(given, SUBJECT, COMMAND_NAME))
    if why:
        return deny(why)
    return allow(f"The analyst hook checked this driver command ({shlex.split(command)[3]}) "
                 f"whole: the one form, run from the repo root, with nothing a shell expands or "
                 f"runs.")


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
