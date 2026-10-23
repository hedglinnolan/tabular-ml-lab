"""The theorist's and the experimentalist's prompts (``claim_cycle/theorist.md``,
``experimentalist.md``): each example command is one the hook lets its role run and the tools read;
the theorist's example proposal is one the validator takes; each prompt names what its own prompt
must carry, its strategy and when it stops; and neither tells the agent which corpus it reads, or
that a copy of the corpus is shuffled or carries planted records."""
from __future__ import annotations

import json
import re
import shlex

import pytest

from research.swarm import vocabulary as V
from research.swarm.claim_cycle import hook as H
from research.swarm.claim_cycle import proposal as P
from research.swarm.claim_cycle import tools as T

PROMPTS = {"theorist": H.HERE / "theorist.md", "experimentalist": H.HERE / "experimentalist.md"}
PREFIX = "scripts/py -m research.swarm.claim_cycle.tools "


def body(role: str) -> str:
    return PROMPTS[role].read_text(encoding="utf-8")


def examples(role: str) -> list[str]:
    return [line.strip() for line in body(role).splitlines()
            if line.strip().startswith(PREFIX) and "<command>" not in line]


@pytest.mark.parametrize("role", list(PROMPTS))
def test_it_shows_one_example_of_each_command_its_role_runs(role):
    assert [e.split()[3] for e in examples(role)] == list(H.COMMANDS[role])


@pytest.mark.parametrize("role, example", [(r, e) for r in PROMPTS for e in examples(r)])
def test_each_example_is_one_the_hook_passes_and_the_tools_read(role, example):
    assert H.bash_refusal(example, H.REPO, role) is None
    args = T.build_parser().parse_args(shlex.split(example)[3:])
    assert args.command in H.COMMANDS[role]


def test_the_theorists_example_proposal_is_one_the_validator_takes():
    (propose,) = [e for e in examples("theorist") if e.split()[3] == "propose"]
    data = json.loads(shlex.split(propose)[-1])
    got = P.validate(data)
    assert (got.status, got.grade) == ("proposed", "unsupported")


@pytest.mark.parametrize("role, names", [
    ("theorist", "the corpus folder, the proposals file and your author id"),
    ("experimentalist", "the corpus folder and the proposals file"),
])
def test_it_names_what_its_prompt_must_carry(role, names):
    text = " ".join(body(role).split())
    assert f"Your prompt must name: {names}." in text
    assert "If any is missing, run nothing and say which." in text


@pytest.mark.parametrize("role", list(PROMPTS))
def test_it_states_its_strategy_and_when_it_stops(role):
    text = body(role)
    assert "\n## How you work\n" in text and "\n## When you stop\n" in text


@pytest.mark.parametrize("role", list(PROMPTS))
def test_it_never_says_which_corpus_it_reads(role):
    text = body(role).lower()
    for word in ("shuffl", "permut", "plant", "sealed file", "derang", "donor", "decoy",
                 "research/corpus", "real corpus", "fake", "synthetic"):
        assert word not in text, word


def test_the_theorist_is_told_every_live_marker_and_no_other_as_a_regions():
    section = body("theorist").split("## The markers", 1)[1].split("\n## ", 1)[0]
    named = set(re.findall(r"`([a-z_]+)`", section))
    assert named == set(V.LIVE)


def test_the_theorist_is_told_each_rule_the_validator_holds_it_to():
    text = " ".join(body("theorist").split())
    for op in P.OPS:
        assert f"`{op}`" in text, op
    for direction in P.DIRECTIONS:
        assert f"`{direction}`" in text, direction
    for kind in P.PROTOCOL:
        assert f"`{kind}`" in text, kind
    for code in P.ENGINE_GAPS:
        assert f"`{code}`" in text, code
    assert f"`{P.SELECTION}`" in text
    assert "not a surfacing claim" in text
    assert "proposed and unsupported" in text


@pytest.mark.parametrize("role", list(PROMPTS))
def test_it_names_what_the_hook_refuses_even_in_quotes(role):
    text = " ".join(body(role).split())
    for c in ("<", ">", "$"):
        assert c in H.FORBIDDEN
    assert 'write "below" for `<`, "above" for `>`, "dollars" for `$`' in text
