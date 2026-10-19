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
  that no shell expands or runs anything in:

  - none of ``;``, ``&``, ``|``, a backtick, ``$``, ``<``, ``>``, a line break or another control
    character anywhere in it (:data:`FORBIDDEN`), quoted or not, so nothing is chained,
    substituted or redirected;
  - outside quotes, only the characters of :data:`PLAIN` and spaces, and ``=`` only inside a word
    after one of them, so no glob, tilde, brace, history, comment, subshell, ``=(...)`` or
    ``=cmd`` of bash's or zsh's, whatever their options, is ever read (zsh is the shell an
    analyst's Bash runs on the owner's machine: its glob qualifiers ``*(e:...:)`` and process
    substitution ``=(...)`` run programs);
  - inside double quotes, no backslash and no ``!``; inside single quotes, anything else.

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
import string
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
QUOTE_IT = (f"Outside quotes a word holds only {SHOWN_PLAIN}, and = inside a word; put anything "
            f"else, such as a reason, in double or single quotes.")


def deny(reason: str) -> dict[str, Any]:
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                   "permissionDecisionReason": reason}}


def _shown(c: str) -> str:
    return {"\n": "a line break", "\r": "a line break", "`": "a backtick", "\t": "a tab",
            "\\": "a backslash"}.get(c) or (f"'{c}'" if c.isprintable() else f"the character U+{ord(c):04X}")


def shell_refusal(command: str) -> str | None:
    """Why a shell could read ``command`` as more than one program's words, or None: the rules in
    the module docstring, checked character by character with the quotes followed as bash and
    zsh follow them."""
    found = [c for c in FORBIDDEN if c in command]
    if found:
        shown = ", ".join(dict.fromkeys(_shown(c) for c in found))
        return (f"The command holds {shown}, which an analyst's command never holds, in a reason "
                f"either: one driver command at a time, nothing chained, substituted or "
                f"redirected. The form is {FORM}.")
    control = next((c for c in command if (ord(c) < 32 or ord(c) == 127) and c != "\t"), None)
    if control is not None:
        return f"The command holds {_shown(control)}, a control character. The form is {FORM}."
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
                    f"run. {QUOTE_IT}")
    if quote is not None:
        return f"The command does not read as one driver command (an unclosed quote?). The form is {FORM}."
    return None


def bash_refusal(command: Any, cwd: Any) -> str | None:
    """Why an analyst's Bash ``command`` is not the driver's invocation, or None when it is."""
    if not isinstance(command, str) or not command.strip():
        return f"An analyst runs only the driver's commands: {FORM}."
    why = shell_refusal(command)
    if why:
        return why
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
