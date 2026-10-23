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
    # Its refits are not counted, as these tools read no sweeps table: null, never a zero.
    assert data["refits"] is None and got.refits.why == P.NOT_COUNTED


def test_the_theorist_writes_refits_not_counted_and_never_a_zero_for_one():
    text = " ".join(body("theorist").split())
    assert "`refits`: write `null`." in text
    assert "never write `0` for a count you do not have" in text


def test_the_survey_counts_records_so_an_effect_on_one_dataset_reaches_it(tmp_path):
    """The theorist's survey and stopping rule count records, not datasets: the shuffled copy's
    planted records all sit on their template's one dataset (``shuffle.planted``), each with its
    decision's materiality measured, so that decision's kind meets the rule on them alone, and it
    is one the theorist may propose about."""
    from research.swarm import shuffle as S
    from research.swarm.claim_cycle import view as W
    from research.swarm.tests import test_shuffle as TS

    text = " ".join(body("theorist").split())
    assert "with a measured materiality in at least two records, on one dataset or several" in text
    assert "every decision kind with a measured materiality in at least two records" in text
    assert "at least two datasets" not in text
    written = S.write(TS.corpus(tmp_path / "corpus"), tmp_path / "copy", tmp_path / "s.json",
                      seed=5, k=6)
    effect, planted = written["planted"]["effect"], set(written["planted"]["records"])
    measured = [v for v in W.read(tmp_path / "copy") if v["run"]["id"] in planted
                and any(d["kind"] == effect["decision"] and d["materiality"]["source"] == "fits"
                        for d in v["decisions"])]
    assert len(measured) == 6 and len({v["dataset"]["id"] for v in measured}) == 1
    assert P.proposable(effect["decision"])


def test_the_experimentalist_ranks_only_by_what_its_tools_serve():
    """The view's refusals and failed fits name a command, a code and a stage, never the decision
    they met, so the pull ranks by the proposal's size, said to stand in for the sweeps' measured
    materiality (SWARM_PLAN §9) until the sweeps are read. A size is in its own loss's units, and
    for `material` it is the smallest difference while for `not_material` the largest (the
    theorist's prompt), so sizes compare only within one loss and one direction: the rank is
    within each, and groups keep the proposals file's order."""
    from research.swarm.claim_cycle import view as W
    from research.swarm.claim_cycle.tests.test_claim_reading import _failed_fit, _refusal, planted

    rec = planted(_failed_fit)
    _refusal(rec)
    shown = W.views([rec])[0]["surfacings"]
    keys = {k for entry in shown["refusals"] + shown["failed_fits"] for k in entry}
    assert keys and not keys & {"decision", "decisions", "seq"}
    text = " ".join(body("experimentalist").split())
    assert "stands in for the sweeps' measured materiality until the sweeps are read" in text
    assert ("Group the proposals by their `loss` (its `purpose` with its `metric` or its "
            "`measure`) and their `direction`, and within each group rank them by `size`, "
            "largest first.") in text
    assert "sizes on different losses or directions do not compare" in text
    assert ("groups are not ranked against one another, and they keep the order in which the "
            "proposals file first names them") in text
    assert "Rank the proposals by `size`, largest first." not in text
    theorist = " ".join(body("theorist").split())
    assert ("for `material` the smallest size of difference among the records you rest it on, "
            "for `not_material` the largest") in theorist
    # The reply says which group a rank is in: its loss and its direction.
    reply = json.loads(body("experimentalist").split("```json", 1)[1].split("```", 1)[0])
    (plan,) = reply["plans"]
    assert {"loss", "predicts", "rank"} <= set(plan)
    assert P.Loss.model_validate(plan["loss"]) and plan["predicts"] in P.DIRECTIONS
    assert "not the decision it met, so they do not rank a proposal" in text
    assert "a decision where a run met a refusal" not in text


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
