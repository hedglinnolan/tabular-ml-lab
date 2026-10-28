"""An analyst run's transcript as the transcript reader reads it (``.claude/agents/
transcript-reader.md``; the owner's ruling of 2026-10-27: an independent cold-start agent, not
the orchestrator, reads the trio's transcripts, against ``research/swarm/transcript_rubric.md``).

    python -m research.swarm.transcript <transcript.jsonl> --out <file.txt>

**What it reads.** Claude Code keeps a subagent's conversation as JSON lines, one object a line:
its ``type`` is ``user`` or ``assistant`` for the conversation's own lines, ``attachment`` for
what the harness adds to it, and something else (a summary) for a record of the session the
analyst is not shown; a conversation line's ``message.content`` is a string or a list of blocks,
``text``, ``thinking``, ``tool_use`` (its ``name`` and ``input``) and ``tool_result`` (its
``content``, a string or a list of blocks, and ``is_error``). A ``user`` line marked ``isMeta`` is
the harness's: a reminder, or a message sent to the analyst while it worked ("The coordinator sent
a message while you were working: ..."). An ``attachment`` line holds its ``attachment`` (its
``type`` and fields) and, where the analyst was shown it, the text it was shown (``rendered``, a
list of ``content`` strings): a queued command among them, such as a coordinator's message or a
background command's notice (its ``prompt``).

**What it writes**, numbered, in the order the lines give them, and nothing else:

* each message to the analyst, its prompt first (a ``user`` line's own string or ``text``
  blocks), headed ``TO THE ANALYST``;
* each thing the harness showed the analyst, headed ``TO THE ANALYST, FROM THE HARNESS``: a
  ``user`` line marked ``isMeta`` (its text), and an attachment not on :data:`HARNESS_CONTEXT`'s
  list (the text it was shown; a queued command with none recorded, its ``prompt``; any other
  with none recorded, its kind and that its text is not recorded);
* each message of the analyst's (an ``assistant`` line's ``text`` block), headed ``THE ANALYST``;
* each command it ran (a ``tool_use`` block): a Bash command's ``command`` as written, another
  tool's name and its input as JSON, headed ``COMMAND``;
* each command's output as the analyst was shown it (its ``tool_result``), headed ``OUTPUT``, or
  ``OUTPUT, MARKED AN ERROR`` where the result is marked so; a block of it that is not text is
  said as one, not shown.

No thinking, no harness context on :data:`HARNESS_CONTEXT`'s list (the same for every run, or
about the session rather than the analysis: the model's name and the account's email address
among it), no record of the session, no command's description, and no usage, id, time or model. A block that a
later line of the same message repeats is written once. A line that is not JSON is refused
(:class:`Unread`), naming its number, and nothing is written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

CONVERSATION = ("user", "assistant")
HEADS = {"to": "TO THE ANALYST", "harness": "TO THE ANALYST, FROM THE HARNESS",
         "analyst": "THE ANALYST", "command": "COMMAND", "output": "OUTPUT",
         "error": "OUTPUT, MARKED AN ERROR"}
# The attachments the extract leaves out, each kind and what it holds, and no other: the harness's
# context, the same for every run or about the session rather than the analysis (the kinds the
# subagent transcripts on the owner's machine held, 2026-10-28). Any other attachment is written.
HARNESS_CONTEXT = {
    "total_tokens_reminder": "the tokens left",
    "date": "today's date",
    "environment": "the working directory, the platform and the shell",
    "model": "the model's name",
    "session_context": "the account's email address and the git status",
    "credential_org": "the account's organization",
    "remote_session_change": "the session's link and the attribution lines for commits",
    "auto_mode": "the permission mode's guidance",
    "instructions": "the repository's CLAUDE.md files",
    "nested_memory": "a folder's CLAUDE.md file",
    "prompt_snapshot": "a copy of the system prompt and the tools",
    "skill_listing": "the skills listed",
    "agent_listing_delta": "the agent types listed",
    "deferred_tools_delta": "the tools listed",
    "deferred_tools_record": "a record of the tools' inputs",
    "mcp_instructions_delta": "the connected servers' instructions",
    "bash_output_audience_note": "that a command's output is the analyst's to read",
    "batching_reminder_sent": "a reminder to make independent calls together",
    "structured_output": "the data of the analyst's own report call, which its command shows",
}


class Unread(ValueError):
    """The transcript cannot be read as JSON lines: where, in words."""


def _lines(path: Path) -> Iterator[dict[str, Any]]:
    for n, text in enumerate(Path(path).read_text("utf-8").splitlines(), start=1):
        if not text.strip():
            continue
        try:
            item = json.loads(text)
        except ValueError:
            raise Unread(f"{path}: line {n} is not JSON, so the transcript is not read.") from None
        if isinstance(item, dict):
            yield item


def _article(word: str) -> str:
    return "an" if word[:1] in "aeiou" else "a"


def _output(content: Any) -> str:
    """A tool result's content as the analyst was shown it: its text, each other block said."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
        else:
            kind = block.get("type") if isinstance(block, dict) else type(block).__name__
            parts.append(f"[{_article(str(kind))} {kind} block, not shown]")
    return "\n".join(parts)


def _command(block: dict[str, Any]) -> str:
    given = block.get("input") or {}
    if block.get("name") == "Bash" and isinstance(given, dict) and "command" in given:
        return str(given["command"])
    return f"{block.get('name')} {json.dumps(given, ensure_ascii=False, sort_keys=True)}"


def _attachment(item: dict[str, Any]) -> str | None:
    """What an attachment line showed the analyst (the module docstring), or None where its kind
    is on :data:`HARNESS_CONTEXT`'s list."""
    attachment = item.get("attachment")
    kind = str(attachment.get("type")) if isinstance(attachment, dict) else "unnamed"
    if kind in HARNESS_CONTEXT:
        return None
    rendered = item.get("rendered")
    if isinstance(rendered, list) and rendered:
        return "\n\n".join(_output(r.get("content")) if isinstance(r, dict) else str(r)
                           for r in rendered)
    if kind == "queued_command" and attachment.get("prompt"):
        return _output(attachment["prompt"])
    return f"[{_article(kind)} {kind} attachment, whose text the transcript does not record]"


def entries(path: Path) -> list[tuple[str, str]]:
    """The transcript's entries (the module docstring), each ``(head, text)``, in order."""
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in _lines(path):
        kind = item.get("type")
        message = item.get("message")
        if kind == "attachment":
            shown = _attachment(item)
            if shown is not None and shown.strip():
                out.append((HEADS["harness"], shown))
            continue
        if kind not in CONVERSATION or not isinstance(message, dict):
            continue
        meta = kind == "user" and bool(item.get("isMeta"))
        content = message.get("content")
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
        mid = message.get("id")
        for block in blocks or []:
            if not isinstance(block, dict):
                continue
            if mid:  # a block a later line of the same message repeats is written once
                key = (str(mid), json.dumps(block, sort_keys=True, default=str))
                if key in seen:
                    continue
                seen.add(key)
            what = block.get("type")
            if what == "text" and str(block.get("text") or "").strip():
                head = "harness" if meta else "to" if kind == "user" else "analyst"
                out.append((HEADS[head], str(block["text"])))
            elif what == "tool_use" and kind == "assistant":
                out.append((HEADS["command"], _command(block)))
            elif what == "tool_result" and kind == "user":
                out.append((HEADS["error" if block.get("is_error") else "output"],
                            _output(block.get("content"))))
    return out


def extract(path: Path) -> str:
    """The transcript at ``path`` as the reader's plain text (the module docstring)."""
    return "\n".join(f"[{i}] {head}\n{text.rstrip()}\n"
                     for i, (head, text) in enumerate(entries(path), start=1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m research.swarm.transcript",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("transcript", type=Path, help="an analyst subagent's JSONL transcript")
    parser.add_argument("--out", type=Path, required=True, help="the plain text file to write")
    return parser


def main(argv: Sequence[str] | None = None, out: Callable[[str], Any] | None = None) -> int:
    say = out or (lambda text: sys.stdout.write(text + "\n"))
    args = build_parser().parse_args(argv)
    try:
        text = extract(args.transcript)
    except (Unread, OSError) as exc:
        say(json.dumps({"refused": str(exc)}))
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, "utf-8")
    say(json.dumps({"written": str(args.out), "entries": len(entries(args.transcript))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
