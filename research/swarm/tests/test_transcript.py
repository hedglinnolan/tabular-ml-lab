"""The transcript reader's extract (``research/swarm/transcript.py``) and its role
(``.claude/agents/transcript-reader.md``, ``research/swarm/transcript_rubric.md``; the owner's
ruling of 2026-10-27: an independent cold-start agent, not the orchestrator, reads the trio's
transcripts).

The extract is checked on a constructed JSONL transcript in Claude Code's shape, its expected text
written here by hand from the construction: each message to the analyst and of the analyst, each
command and each output, in order, and nothing else (no thinking, no attachment the harness added,
no meta line, no usage, ids, times or model, no command's description, a block repeated under one
message id once)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.swarm import transcript as T

REPO = Path(__file__).resolve().parents[3]
ROLE = REPO / ".claude" / "agents" / "transcript-reader.md"
RUBRIC = REPO / "research" / "swarm" / "transcript_rubric.md"

ASK = "scripts/py -m research.swarm.driver ask --run run_01-" + "a" * 32
ANSWER = ("scripts/py -m research.swarm.driver answer --run run_01-" + "a" * 32
          + ' impute --marker missingness --reason "Blanks in x."')


def line(kind: str, content, **more) -> dict:
    return {"type": kind, "uuid": "u", "timestamp": "2026-10-28T10:00:00Z", "isSidechain": True,
            "message": {"role": kind, "content": content, **more.pop("message", {})}, **more}


def constructed() -> list[dict]:
    usage = {"input_tokens": 3, "output_tokens": 9}
    return [
        line("user", "Run run_01. The card: does x lower y?"),
        {"type": "attachment", "attachment": {"type": "reminder", "content": "SECRET reminder"}},
        line("assistant", [{"type": "thinking", "thinking": "SECRET thought", "signature": "s"}],
             message={"id": "m1", "model": "claude-haiku-5-5", "usage": usage}),
        line("assistant", [{"type": "text", "text": "I will ask first."}],
             message={"id": "m1", "model": "claude-haiku-5-5", "usage": usage}),
        line("assistant", [{"type": "tool_use", "id": "t1", "name": "Bash",
                            "input": {"command": ASK, "description": "SECRET description"}}],
             message={"id": "m1", "usage": usage}),
        # The same block again under the same message id (a line that repeats it): once.
        line("assistant", [{"type": "tool_use", "id": "t1", "name": "Bash",
                            "input": {"command": ASK, "description": "SECRET description"}}],
             message={"id": "m1", "usage": usage}),
        line("user", [{"type": "tool_result", "tool_use_id": "t1", "content": '{"question": 1}',
                       "is_error": False}], toolUseResult={"stdout": "SECRET duplicate"}),
        line("user", "SECRET caveat", isMeta=True),
        {"type": "summary", "summary": "SECRET summary"},
        line("assistant", [{"type": "text", "text": "Now the answer."},
                           {"type": "tool_use", "id": "t2", "name": "Bash",
                            "input": {"command": ANSWER}}],
             message={"id": "m2", "usage": usage}),
        line("user", [{"type": "tool_result", "tool_use_id": "t2", "is_error": True,
                       "content": [{"type": "text", "text": "Refused by the hook."},
                                   {"type": "image", "source": {"data": "SECRET"}}]}]),
        line("assistant", [{"type": "tool_use", "id": "t3", "name": "Read",
                            "input": {"file_path": "x.txt"}}], message={"id": "m3"}),
        line("user", [{"type": "text", "text": "Stop now."}]),
    ]


EXPECTED = f"""[1] TO THE ANALYST
Run run_01. The card: does x lower y?

[2] THE ANALYST
I will ask first.

[3] COMMAND
{ASK}

[4] OUTPUT
{{"question": 1}}

[5] THE ANALYST
Now the answer.

[6] COMMAND
{ANSWER}

[7] OUTPUT, MARKED AN ERROR
Refused by the hook.
[an image block, not shown]

[8] COMMAND
Read {{"file_path": "x.txt"}}

[9] TO THE ANALYST
Stop now.
"""


def write_jsonl(path: Path, lines: list[dict]) -> Path:
    path.write_text("".join(json.dumps(x) + "\n" for x in lines) + "\n", "utf-8")
    return path


def test_the_extract_holds_the_messages_commands_and_outputs_in_order_and_nothing_else(tmp_path):
    path = write_jsonl(tmp_path / "agent.jsonl", constructed())
    text = T.extract(path)
    assert text == EXPECTED
    assert "SECRET" not in text and "claude-haiku" not in text and "2026-10-28" not in text


def test_the_command_writes_the_extract_and_counts_its_entries(tmp_path):
    path = write_jsonl(tmp_path / "agent.jsonl", constructed())
    out = tmp_path / "reader" / "transcript.txt"
    printed: list[str] = []
    assert T.main([str(path), "--out", str(out)], out=printed.append) == 0
    assert out.read_text("utf-8") == EXPECTED
    assert json.loads(printed[0]) == {"written": str(out), "entries": 9}


def test_a_line_that_is_not_json_is_refused_naming_its_number(tmp_path):
    path = tmp_path / "agent.jsonl"
    path.write_text(json.dumps(constructed()[0]) + "\n{not json\n", "utf-8")
    with pytest.raises(T.Unread, match="line 2"):
        T.extract(path)
    printed: list[str] = []
    assert T.main([str(path), "--out", str(tmp_path / "t.txt")], out=printed.append) == 1
    assert "line 2" in json.loads(printed[0])["refused"] and not (tmp_path / "t.txt").exists()


# ── the reader's role ────────────────────────────────────────────────────────


def parts() -> tuple[dict[str, str], str]:
    text = ROLE.read_text("utf-8")
    assert text.startswith("---\n")
    head, body = text[4:].split("\n---\n", 1)
    fields = {}
    for row in head.splitlines():
        name, _, value = row.partition(":")
        fields[name.strip()] = value.strip()
    return fields, body


def test_the_reader_is_read_only_and_its_first_lines_name_what_its_prompt_must():
    fields, body = parts()
    assert fields["name"] == "transcript-reader"
    assert [t.strip() for t in fields["tools"].split(",")] == ["Read", "Grep", "Bash"]
    disallowed = [t.strip() for t in fields["disallowedTools"].split(",")]
    assert {"Edit", "Write", "NotebookEdit", "WebFetch", "WebSearch", "Agent"} <= set(disallowed)
    first = " ".join(body.strip().split("\n\n", 1)[0].split())
    assert first.startswith("Your prompt must name: the transcript file, the run's dataset and "
                            "its purpose.")


def test_the_reader_runs_the_extractor_as_its_command_line_reads_and_reads_the_rubric():
    import shlex

    _, body = parts()
    shown = [row.strip() for row in body.splitlines()
             if row.strip().startswith("scripts/py -m research.swarm.transcript ")]
    assert shown, "the role shows the extractor's command"
    for command in shown:
        T.build_parser().parse_args(shlex.split(command)[3:])
    assert "research/swarm/transcript_rubric.md" in body and RUBRIC.is_file()
    text = " ".join(body.split())
    for never in ("research/swarm/prereg.json", "research/swarm/generators.py",
                  "research/datasets/", "research/corpus/"):
        assert never in text, never  # each named among what it never reads


def test_the_rubric_names_each_class_of_finding_and_what_could_not_be_judged():
    text = " ".join(RUBRIC.read_text("utf-8").split())
    for heading in ("Misreadings", "Refusals ignored", "Ways forward misused or not taken",
                    "Decisions made without exploring", "Annotations that do not match",
                    "What could not be judged"):
        assert f"## {heading}" in RUBRIC.read_text("utf-8"), heading
    assert "a swarm that makes exactly one decision" in text
    assert "quote" in text.lower()
