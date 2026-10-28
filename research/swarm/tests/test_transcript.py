"""The transcript reader's extract (``research/swarm/transcript.py``) and its role
(``.claude/agents/transcript-reader.md``, ``research/swarm/transcript_rubric.md``; the owner's
ruling of 2026-10-27: an independent cold-start agent, not the orchestrator, reads the trio's
transcripts).

The extract is checked on a constructed JSONL transcript whose lines have the shape Claude Code
writes (each line's keys, an attachment's ``rendered`` text and its role, a user line marked
``isMeta``, as in the subagent transcripts on the owner's machine), its contents made up and its
expected text written here by hand from the construction: each message to the analyst and of the
analyst, each command and each output, and what the harness showed the analyst (a coordinator's
message among them, in both of the forms the transcripts hold it), in order, and nothing else (no
thinking, no harness context on the extract's list, no usage, ids, times or model, no command's
description, a block repeated under one message id once)."""
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
# A line's own keys, as Claude Code writes them in a subagent's transcript (made-up values).
COMMON = {"parentUuid": "p", "isSidechain": True, "agentId": "a1", "uuid": "u",
          "timestamp": "2026-10-28T10:00:00Z", "userType": "external", "entrypoint": "cli",
          "cwd": "/SECRET/cwd", "sessionId": "SECRET-session", "version": "2.1.289",
          "gitBranch": "SECRET-branch", "slug": "SECRET-slug"}
COORDINATOR = "Answer the missing values with the median first."
REMINDER = ("<system-reminder id=\"r1\">\nThe coordinator sent a message while you were working:\n"
            f"{COORDINATOR}\n\nAddress this before completing your current task.\n"
            "</system-reminder id=\"r1\">")
HANDBACK = ("<system-reminder>\nYour final report is delivered through SubagentHandback: call it "
            "last.\n</system-reminder>")
CHANGED = ("<system-reminder>\nNote: x.txt changed on disk since you last read it.\n"
           "</system-reminder>")


def line(kind: str, content, **more) -> dict:
    return {**COMMON, "promptId": "q", "type": kind,
            "message": {"role": kind, "content": content, **more.pop("message", {})}, **more}


def attached(kind: str, rendered: str | None = None, role: str = "system", **fields) -> dict:
    """An attachment line: its ``attachment`` (``kind`` and ``fields``) and, where the harness
    showed it, the text it showed (``rendered``, in ``role``'s place)."""
    out = {**COMMON, "type": "attachment", "attachment": {"type": kind, **fields}}
    if rendered is not None:
        out |= {"rendered": [{"content": rendered}], "renderedRole": role}
    return out


def harness_context() -> list[dict]:
    """One attachment of each kind the extract leaves out (:data:`transcript.HARNESS_CONTEXT`),
    each in the shape the transcripts hold it, its text marked SECRET."""
    shown = "<system-reminder>\nSECRET {}\n</system-reminder>"
    return [
        attached("total_tokens_reminder", shown.format("tokens"),
                 text="<total_tokens>SECRET</total_tokens>"),
        attached("date", shown.format("date"), date="2026-10-28"),
        attached("environment", shown.format("environment"), snapshot={"platform": "SECRET"}),
        attached("model", shown.format("You are powered by the model named Haiku 5.5. The exact "
                                       "model ID is claude-haiku-5-5."),
                 identity={"modelId": "claude-haiku-5-5"}, text="SECRET model"),
        attached("session_context", shown.format("email research@example.org"), role="user",
                 context={"userEmail": "SECRET"}),
        attached("credential_org", organizationUuid="SECRET"),
        attached("remote_session_change", shown.format("attribution"), role="user",
                 url="SECRET"),
        attached("auto_mode", shown.format("auto mode"), bypass=False),
        attached("instructions", shown.format("CLAUDE.md"), role="user", files=[]),
        attached("nested_memory", shown.format("frontend CLAUDE.md"), path="SECRET",
                 content={"content": "SECRET"}),
        attached("prompt_snapshot", systemPrompt=["SECRET system prompt"]),
        attached("skill_listing", shown.format("skills"), content="SECRET skills"),
        attached("agent_listing_delta", shown.format("agents"), addedTypes=["analyst"]),
        attached("deferred_tools_delta", shown.format("tools"), addedNames=["SECRET"]),
        attached("deferred_tools_record", entries=[]),
        attached("mcp_instructions_delta", shown.format("servers"), addedNames=["SECRET"]),
        attached("bash_output_audience_note", shown.format("only you see it"), toolUseID="t1"),
        attached("batching_reminder_sent", text="SECRET batching", model="claude-haiku-5-5"),
        attached("structured_output", data={"SECRET": 1}, toolUseID="t9"),
    ]


def constructed() -> list[dict]:
    usage = {"input_tokens": 3, "output_tokens": 9}
    return [
        line("user", "Run run_01. The card: does x lower y?"),
        *harness_context()[:6],
        line("user", HANDBACK, isMeta=True),
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
        *harness_context()[6:],
        # A coordinator's message as a queued command the harness showed (its rendered text) ...
        attached("queued_command", REMINDER, prompt=COORDINATOR, source_uuid="s",
                 origin={"kind": "coordinator"}, isMeta=True, reminderId="r1"),
        # ... and as a user line marked isMeta.
        line("user", f"The coordinator sent a message while you were working:\n{COORDINATOR}",
             isMeta=True),
        {"type": "summary", "summary": "SECRET summary"},
        line("assistant", [{"type": "text", "text": "Now the answer."},
                           {"type": "tool_use", "id": "t2", "name": "Bash",
                            "input": {"command": ANSWER}}],
             message={"id": "m2", "usage": usage}),
        line("user", [{"type": "tool_result", "tool_use_id": "t2", "is_error": True,
                       "content": [{"type": "text", "text": "Refused by the hook."},
                                   {"type": "image", "source": {"data": "SECRET"}}]}]),
        attached("edited_text_file", CHANGED, filename="x.txt", snippet=""),
        # A queued command with no text shown recorded: its prompt.
        attached("queued_command", prompt="<task-notification>done</task-notification>",
                 origin={"kind": "task-notification"}, commandMode="task-notification"),
        # An attachment of a kind the extract does not know, with no text shown recorded.
        attached("a_new_kind", payload="SECRET payload"),
        line("assistant", [{"type": "tool_use", "id": "t3", "name": "Read",
                            "input": {"file_path": "x.txt"}}], message={"id": "m3"}),
        line("user", [{"type": "text", "text": "Stop now."}]),
    ]


EXPECTED = f"""[1] TO THE ANALYST
Run run_01. The card: does x lower y?

[2] TO THE ANALYST, FROM THE HARNESS
{HANDBACK}

[3] THE ANALYST
I will ask first.

[4] COMMAND
{ASK}

[5] OUTPUT
{{"question": 1}}

[6] TO THE ANALYST, FROM THE HARNESS
{REMINDER}

[7] TO THE ANALYST, FROM THE HARNESS
The coordinator sent a message while you were working:
{COORDINATOR}

[8] THE ANALYST
Now the answer.

[9] COMMAND
{ANSWER}

[10] OUTPUT, MARKED AN ERROR
Refused by the hook.
[an image block, not shown]

[11] TO THE ANALYST, FROM THE HARNESS
{CHANGED}

[12] TO THE ANALYST, FROM THE HARNESS
<task-notification>done</task-notification>

[13] TO THE ANALYST, FROM THE HARNESS
[an a_new_kind attachment, whose text the transcript does not record]

[14] COMMAND
Read {{"file_path": "x.txt"}}

[15] TO THE ANALYST
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
    assert "research@example.org" not in text and "Haiku" not in text


def test_the_harness_context_left_out_is_its_list_and_a_coordinators_message_is_not_on_it():
    """The constructed transcript holds one attachment of each kind on the list, and each is left
    out; the kinds that carry a message to the analyst (a queued command, a file changed on disk)
    are not on it."""
    kinds = [a["attachment"]["type"] for a in harness_context()]
    assert sorted(kinds) == sorted(T.HARNESS_CONTEXT)
    assert not {"queued_command", "edited_text_file"} & set(T.HARNESS_CONTEXT)


def test_the_command_writes_the_extract_and_counts_its_entries(tmp_path):
    path = write_jsonl(tmp_path / "agent.jsonl", constructed())
    out = tmp_path / "reader" / "transcript.txt"
    printed: list[str] = []
    assert T.main([str(path), "--out", str(out)], out=printed.append) == 0
    assert out.read_text("utf-8") == EXPECTED
    assert json.loads(printed[0]) == {"written": str(out), "entries": 15}


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
    for head in T.HEADS.values():  # each heading the extract writes, described
        assert f"`{head}`" in text, head
