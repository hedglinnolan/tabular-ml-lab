"""The analyst agent type (``.claude/agents/analyst.md``, DRIVER_PLAN D2): Bash its only tool, the
listed tools disallowed, Haiku at high effort; each command it is shown is one the hook lets
through and the driver reads; and it carries no materiality field and no known answer."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from research.swarm import analyst_hook, driver

REPO = Path(__file__).resolve().parents[3]
AGENT = REPO / ".claude" / "agents" / "analyst.md"


def parts() -> tuple[dict[str, str], str]:
    text = AGENT.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    head, body = text[4:].split("\n---\n", 1)
    fields = {}
    for line in head.splitlines():
        name, _, value = line.partition(":")
        fields[name.strip()] = value.strip()
    return fields, body


def test_its_definition_holds_it_to_bash():
    fields, _ = parts()
    assert fields["name"] == "analyst"
    assert fields["tools"] == "Bash"
    disallowed = [t.strip() for t in fields["disallowedTools"].split(",")]
    assert disallowed == ["Edit", "Write", "NotebookEdit", "WebFetch", "WebSearch", "Agent", "mcp__*"]
    assert fields["model"] == "haiku" and fields["effort"] == "high"


def examples() -> list[str]:
    _, body = parts()
    return [line.strip() for line in body.splitlines()
            if line.strip().startswith("scripts/py -m research.swarm.driver ")
            and "<command>" not in line]


def test_it_shows_one_example_of_each_command():
    shown = [e.split()[3] for e in examples()]
    assert shown == list(driver.ANALYST_COMMANDS)


@pytest.mark.parametrize("example", examples())
def test_each_example_is_one_the_hook_passes_and_the_driver_reads(example):
    assert analyst_hook.bash_refusal(example, REPO) is None
    import shlex

    args = driver.build_parser().parse_args(shlex.split(example)[3:])
    assert args.command in driver.ANALYST_COMMANDS and args.run == "RUN"


def test_it_names_what_its_prompt_must_carry_and_states_no_answer():
    _, body = parts()
    assert "Your prompt must name: the run id, and the question card" in body
    for word in ("materiality", "validity", "known answer", "FIXTURE_TRUTHS", "truths",
                 "notes/replication", "corpus", "referee"):
        assert word.lower() not in body.lower(), word
    assert not re.search(r"\b\d+(\.\d+)?%", body)  # no number it could take for a result


def test_it_says_what_a_refusal_lists_as_the_driver_prints_it():
    """DRV-INT V1 (the verifier, round 2): the prompt's account of a refusal is the driver's. A
    refusal of the driver's own prints no ways forward, and a way forward the app gives with no
    decision has no id, which ``answer`` cannot name; each is read here from the driver itself."""
    import json

    from research.swarm import options as opts

    out: list[str] = []
    assert driver.main(["ask", "--run", "r1"], out=out.append) == 1
    own = json.loads(out[0])["refused"]
    assert own["by"] == "driver" and own["message"] and "exits" not in own
    wait = {"label": "Wait for the fit to finish", "decision": None}
    take = {"label": "Confirm", "decision": {"kind": "confirm_reading", "reading": "code_or_count",
                                             "column": "age", "value": "amount"}}
    ways = opts.exits([wait, take])
    assert ways[0]["id"] is None and ways[1]["id"].startswith("x-")
    assert opts.resolve({"options": []}, [ways[1]["id"]], [], ways) == take["decision"]
    with pytest.raises(opts.Unanswerable):
        opts.resolve({"options": []}, ["None"], [], ways)

    _, body = parts()
    text = " ".join(body.split())
    assert "lists the ways forward, each with an id" not in text
    assert 'A refusal by the driver (`"by": "driver"`) lists none, and its `message` says why.' in text
    assert ('A refusal by the app (`"by": "server"`) lists its ways forward under `exits`.' in text)
    assert ("A way forward with an id, in a refusal or in `ask`, is taken with "
            "`answer <id> --reason <why>`; one whose id is null records nothing, and `answer` "
            "cannot take it.") in text
