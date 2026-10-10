"""SURFACING_POLICY recommendation 2 (§1.6): ``fires``, ``reads`` and ``holds`` are executable for
every item the quest log can surface (``turbotab/core/surfacing.py``), and the crosswalk names the
predicate of each item that has one.

The expectations come from outside the registry: the Router's own steps (``interview.route``), the
quest log's lines (``quest.quest_log``), the stage graph's ``requires``, the decision union and
``crosswalk.json``. The crosswalk items whose ``fires_when`` is still prose are counted UNWIRED,
and the count is recorded here as it is, so wiring one lowers it and nothing raises it unseen.

Run as a script with ``--write-rules`` to write each wired item's ``rule`` into
``crosswalk.json``.
"""
from __future__ import annotations

import importlib
import json
import sys
import typing
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pytest

from turbotab.core import decisions, quest, surfacing
from turbotab.core.decisions import ProjectState
from turbotab.core.interview import QUESTION_KEYS, route
from turbotab.core.quest import Facts
from turbotab.core.tests.path_fuzzer import FIXTURES, artifacts_for, question_options

REPO = Path(__file__).resolve().parents[3]
CROSSWALK = REPO / "docs" / "turbotab-next" / "crosswalk" / "crosswalk.json"
SEVEN = ("data", "question", "first_look", "whos_in", "models", "results", "writeup")
# The crosswalk items whose fires_when is still prose (no ``rule``), counted on 2026-10-09: 106 of
# the 708 are wired (the Router's questions, every decision kind, the engine's findings and
# Explore's, the compute stages' results and the defaults the engine states). The 602 left are
# 271 catalog threads, 51 refusals, 49 noticings, 46 defaults, 42 exhibits, 33 previews,
# 27 decisions with no engine kind yet (stacking, the designed experiments, recipes and tuning),
# 23 other, 17 sentinels, 16 exports, 10 results and 17 more.
UNWIRED = 602


@lru_cache(maxsize=1)
def crosswalk() -> dict:
    return json.loads(CROSSWALK.read_text("utf-8"))


def decision_kinds() -> set[str]:
    union = typing.get_args(typing.get_args(decisions.Decision)[0])
    return {model.model_fields["kind"].default for model in union}


def states() -> list[tuple[ProjectState, Facts]]:
    """A spread of states over both fixtures: each prefix of a journey answering the first option
    of every open question, with the artifacts the fuzzer mocks."""
    out = []
    for fx in FIXTURES:
        for purpose in ("inference", "prediction"):
            state, records = ProjectState(), []
            for _ in range(len(QUESTION_KEYS) + 4):
                artifacts = artifacts_for(state, fx, records)
                out.append((state, Facts(columns=fx.columns, artifacts=artifacts)))
                steps = route(state, {"fit": {"status": "idle"}}, artifacts, records)
                first = next((s for s in steps if s.status == "open"), None)
                options = question_options(first.key, state, fx) if first is not None else []
                if first is not None and first.key == "purpose":
                    options = [{"kind": "set_purpose", "purpose": purpose}]
                if not options:
                    break
                decision = decisions.validate(options[0], {"columns": list(fx.columns)})
                state = decisions.fold_onto(state, decision)
    return out


# ── every engine object has an executable declaration ───────────────────────


def test_every_router_key_decision_kind_finding_and_result_has_an_executable_declaration():
    registry = surfacing.registry()
    assert {k for k in registry if k.startswith("question:")} == {
        f"question:{k}" for k in QUESTION_KEYS}
    # Every kind the log accepts; a revert is the record it undoes (quest.FOLLOWS_WHAT_IT_UNDOES).
    for kind in decision_kinds() - {quest.FOLLOWS_WHAT_IT_UNDOES}:
        assert surfacing.for_kind(kind).shape == "decision", kind
    from turbotab.core.stages.finding_words import FAMILIES

    for fam in FAMILIES:
        assert f"noticing:{fam}" in registry, fam
    for kind in quest.EXPLORE_FINDINGS:
        assert f"noticing:explore::{kind}" in registry, kind
    assert {k for k in registry if k.startswith("result:")} == {
        f"result:{name}" for name in quest.COMPUTE}
    graph = {s.name for s in surfacing._order()}
    known = set(QUESTION_KEYS) | {d.kind for d in quest.DECLARATIONS}
    empty = ProjectState()
    for key, item in registry.items():
        assert isinstance(item, surfacing.Surfaceable), key
        assert item.shape in surfacing.SHAPES and item.stage in SEVEN, key
        assert isinstance(item.fires(empty, Facts()), bool), key
        assert set(item.reads) <= known, (key, set(item.reads) - known)
        assert item.consumer is None or item.consumer in graph, key
        if item.shape in ("question", "decision") and not key.startswith("decision:confirm_sweep"):
            assert item.holds, key


def test_a_question_fires_exactly_where_the_router_asks_states_or_answers_it():
    seen = Counter()
    for state, facts in states():
        steps = {s.key: s.status for s in route(state, {}, facts.artifacts)}
        for key in QUESTION_KEYS:
            fires = surfacing.registry()[f"question:{key}"].fires(state, facts)
            assert fires == (steps[key] != "not_applicable"), (key, steps[key], state)
            seen[fires] += 1
    assert seen[True] and seen[False]


def test_a_declaration_fires_exactly_where_the_quest_log_lists_it():
    listed_somewhere = set()
    for state, facts in states():
        steps = route(state, {}, facts.artifacts)
        log = quest.quest_log(state, [], steps, columns=facts.columns, artifacts=facts.artifacts)
        listed = {l.key for s in log.stages for l in s.lines if l.source == "declaration"}
        for decl in quest.DECLARATIONS:
            if decl.only_recorded:
                continue
            fires = surfacing.for_kind(decl.kind).fires(state, facts)
            assert fires == (decl.kind in listed), (decl.kind, state)
            if fires:
                listed_somewhere.add(decl.kind)
    assert {"set_batch", "set_selection", "set_model_sequence", "set_intended_use"} <= listed_somewhere


def test_holds_names_every_slot_an_answer_changes():
    for state, facts in states():
        fx = FIXTURES[0] if "dietary" in (state.lens or ()) else FIXTURES[1]
        for key in QUESTION_KEYS:
            for payload in question_options(key, state, fx):
                try:
                    decision = decisions.validate(payload, {"columns": list(fx.columns)})
                except Exception:
                    continue
                after = decisions.fold_onto(state, decision)
                changed = {n for n in ProjectState.model_fields
                           if getattr(after, n) != getattr(state, n)}
                holds = surfacing.for_kind(decision.kind).holds | quest.written_slots(decision)
                assert changed <= holds, (decision.kind, changed - holds)
                assert decisions.SLOTS[decision.kind] in surfacing.for_kind(decision.kind).holds


def test_a_result_fires_when_the_graph_can_compute_it():
    from turbotab.core.estimand import ESTIMATE_STAGES

    registry = surfacing.registry()
    nothing = ProjectState(lens=["dietary"], target="glucose")
    # With no goal nothing estimated computes (an unanswered purpose is the strictest case).
    assert not any(registry[f"result:{s}"].fires(nothing, Facts()) for s in ESTIMATE_STAGES
                   if f"result:{s}" in registry)
    # The outcome alone (Explore) waits for the split, the last of Who's in.
    assert not registry["result:explore"].fires(nothing, Facts())
    split = nothing.model_copy(update={"split": decisions.SplitSpec(holdout=0.0)})
    assert registry["result:explore"].fires(split, Facts())
    assert "explore" in surfacing.computable(split)
    assert registry["result:explore"].reads == ("lens", "target", "split")


def _journey_states(n: int = 60, every: int = 2) -> list[tuple[ProjectState, dict]]:
    from turbotab.core.tests.path_fuzzer import run_journey

    return [(snap.state, snap.artifacts) for seed in range(n)
            for snap in run_journey(seed).snapshots[::every]]


def test_a_question_reads_the_answers_its_gate_needs_and_not_every_earlier_one():
    # reads names what the question needs (the task: "the inputs it needs"), not the asking order:
    # clearing every answer a question does not read leaves the Router's gate where it was (asked,
    # stated or not applicable), over the fuzzer's states on both fixtures.
    from turbotab.core.interview import applicability, stated

    registry = surfacing.registry()
    holds = {k: registry[f"question:{k}"].holds for k in QUESTION_KEYS}
    fields = ProjectState.model_fields
    checked = 0
    for state, artifacts in _journey_states():
        for key in QUESTION_KEYS:
            reads = registry[f"question:{key}"].reads
            kept = set().union(*(holds[r] for r in (*reads, key)))
            cleared = {slot: None for other in QUESTION_KEYS if other not in (*reads, key)
                       for slot in holds[other] - kept
                       if slot in fields and getattr(state, slot) is not None}
            if not cleared:
                continue
            bare = state.model_copy(update=cleared)
            before = (applicability(key, state, artifacts), stated(key, state, artifacts))
            after = (applicability(key, bare, artifacts), stated(key, bare, artifacts))
            assert (before[0] is None, before[1] is None) == (after[0] is None, after[1] is None), (
                key, sorted(cleared), before, after)
            checked += 1
    assert checked > 10_000
    # Each read is an earlier question (the display-order rule), and most questions read far
    # fewer than every earlier one.
    for i, key in enumerate(QUESTION_KEYS):
        reads = registry[f"question:{key}"].reads
        assert set(reads) <= set(QUESTION_KEYS[:i]), (key, reads)
    assert registry["question:open_seal"].reads == ("split", "models")
    assert registry["question:clusters"].reads == ("roles",)
    # A decision kind answering a question reads what that question reads.
    assert surfacing.for_kind("set_censoring").reads == ("target", "task")


def test_the_registry_blocks_a_result_exactly_where_the_engine_does(tmp_path):
    # surfacing.blocked reads the scheduler's own rule; the engine's keys, computed as the engine
    # computes them, agree on every state the fuzzer reaches.
    from turbotab.core import graph

    engine = object.__new__(graph.Engine)
    engine._order = quest._graph().order()
    compared = 0
    for state, _artifacts in _journey_states(n=40, every=3):
        project = graph._Project("p")
        project.ctx = graph.ProjectContext(project_id="p", state=state, cache_root=tmp_path,
                                           paths={}, settings={}, fingerprint="f")
        engine._compute_keys(project)
        blocked = surfacing.blocked(state)
        assert {n: tuple(m) for n, m in project.missing.items()} == blocked
        assert {n for n, k in project.keys.items() if k is not None} == surfacing.computable(state)
        compared += 1
    assert compared > 300


# ── the crosswalk's rules ────────────────────────────────────────────────────


def test_each_wired_crosswalk_item_names_its_rule_and_the_rule_resolves():
    items = {i["id"]: i for i in crosswalk()["items"]}
    expected = surfacing.crosswalk_rules()
    on_disk = {i: item["rule"] for i, item in items.items() if "rule" in item}
    assert on_disk == expected, (
        "crosswalk.json's rules drifted from the registry: run "
        "`python -m turbotab.core.tests.test_surfaceable --write-rules`")
    by_item = surfacing.carriers()
    for item_id, rule in expected.items():
        assert items[item_id]["stage"] == surfacing.registry()[by_item[item_id][0]].stage
        module, name = rule.split(":")
        fn = getattr(importlib.import_module(module), name)
        assert callable(fn) and fn.__name__ == name
        for state, facts in states()[::7]:
            assert fn(state, facts) == any(surfacing.registry()[k].fires(state, facts)
                                           for k in by_item[item_id])


def test_a_crosswalk_item_whose_fires_when_is_still_prose_counts_as_unwired():
    items = crosswalk()["items"]
    unwired = [i for i in items if not i.get("rule")]
    by_prefix = Counter(i["id"].split(":", 1)[0] for i in unwired)
    print(f"\nUNWIRED: {len(unwired)} of {len(items)} crosswalk items "
          f"({len(items) - len(unwired)} wired): {dict(by_prefix.most_common())}")
    assert len(unwired) == UNWIRED, (
        f"{len(unwired)} crosswalk items are UNWIRED, not {UNWIRED}: update UNWIRED (and its "
        f"comment) to the count as it is")
    assert "rule" in crosswalk()["schema"]


# ── modes ────────────────────────────────────────────────────────────────────


def test_a_mode_changes_the_level_a_line_is_drawn_at_never_its_label():
    state, facts = states()[3]
    steps = route(state, {}, facts.artifacts)
    log = quest.quest_log(state, [], steps, columns=facts.columns, artifacts=facts.artifacts)
    standard = surfacing.disclosure(log, "standard")
    opened = [s for s in standard if s.level == 3]
    assert len(opened) == 1
    expert = surfacing.disclosure(log, "knows_the_field", remembered=(opened[0].key,))
    assert [(s.id, s.key, s.label) for s in expert] == [(s.id, s.key, s.label) for s in standard]
    assert not any(s.level == 3 for s in expert)
    with pytest.raises(ValueError):
        surfacing.disclosure(log, "speedy")


# ── the writer ───────────────────────────────────────────────────────────────


def _write_rules() -> None:
    data = json.loads(CROSSWALK.read_text("utf-8"))
    rules = surfacing.crosswalk_rules()
    data["schema"]["rule"] = (
        "module:function, the executable predicate of fires_when (turbotab/core/surfacing.py): "
        "true when an engine object the item carries fires. Absent while fires_when is prose; "
        "the registry test counts such an item as UNWIRED.")
    items = []
    for item in data["items"]:
        out = {}
        for key, value in item.items():
            if key == "rule":
                continue
            out[key] = value
            if key == "fires_when" and item["id"] in rules:
                out["rule"] = rules[item["id"]]
        items.append(out)
    data["items"] = items
    CROSSWALK.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(rules)} rules into {CROSSWALK}", file=sys.stderr)


if __name__ == "__main__":
    if "--write-rules" in sys.argv[1:]:
        _write_rules()
