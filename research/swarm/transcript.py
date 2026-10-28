"""An analyst run's transcript as the transcript reader reads it (``.claude/agents/
transcript-reader.md``; the owner's ruling of 2026-10-27: an independent cold-start agent, not
the orchestrator, reads the trio's transcripts, against ``research/swarm/transcript_rubric.md``).

    python -m research.swarm.transcript <transcript.jsonl> --out <file.txt>

**What it reads.** Claude Code keeps a subagent's conversation as JSON lines, one object a line:
its ``type`` is ``user`` or ``assistant`` for the conversation's own lines, and something else
(an attachment, a summary) for what the harness adds; a conversation line's ``message.content`` is
a string or a list of blocks, ``text``, ``thinking``, ``tool_use`` (its ``name`` and ``input``)
and ``tool_result`` (its ``content``, a string or a list of blocks, and ``is_error``). A ``user``
line marked ``isMeta`` is the harness's too.

**What it writes**, numbered, in the order the lines give them, and nothing else:

* each message to the analyst, its prompt first (a ``user`` line's own string or ``text``
  blocks), headed ``TO THE ANALYST``;
* each message of the analyst's (an ``assistant`` line's ``text`` block), headed ``THE ANALYST``;
* each command it ran (a ``tool_use`` block): a Bash command's ``command`` as written, another
  tool's name and its input as JSON, headed ``COMMAND``;
* each command's output as the analyst was shown it (its ``tool_result``), headed ``OUTPUT``, or
  ``OUTPUT, MARKED AN ERROR`` where the result is marked so; a block of it that is not text is
  said as one, not shown.

No thinking, no line the harness added, no command's description, and no usage, id, time or
model. A block that a later line of the same message repeats is written once. A line that is not
JSON is refused (:class:`Unread`), naming its number, and nothing is written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

CONVERSATION = ("user", "assistant")
HEADS = {"to": "TO THE ANALYST", "analyst": "THE ANALYST", "command": "COMMAND",
         "output": "OUTPUT", "error": "OUTPUT, MARKED AN ERROR"}


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
            parts.append(f"[an {kind} block, not shown]" if str(kind)[:1] in "aeiou"
                         else f"[a {kind} block, not shown]")
    return "\n".join(parts)


def _command(block: dict[str, Any]) -> str:
    given = block.get("input") or {}
    if block.get("name") == "Bash" and isinstance(given, dict) and "command" in given:
        return str(given["command"])
    return f"{block.get('name')} {json.dumps(given, ensure_ascii=False, sort_keys=True)}"


def entries(path: Path) -> list[tuple[str, str]]:
    """The transcript's entries (the module docstring), each ``(head, text)``, in order."""
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in _lines(path):
        kind = item.get("type")
        message = item.get("message")
        if kind not in CONVERSATION or not isinstance(message, dict) or item.get("isMeta"):
            continue
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
                out.append((HEADS["to" if kind == "user" else "analyst"], str(block["text"])))
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
