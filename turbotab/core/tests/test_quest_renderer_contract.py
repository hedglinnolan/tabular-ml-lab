"""W1 part 2 · the renderer contract (plans/W1_PART2_PLAN.md §2 "W1-E", ruling 6): every quest line
the engine emits has a card in the quest shell, never a blank.

This test walks the path fuzzer's sample datasets (the NHANES fixture under the dietary lens, an
untargeted metabolomics table) under every goal, and writes each ``(source, key)`` it meets, with
the stages, labels and dataset/goal pairs it was seen in, to the shell's fixture
``turbotab/frontend/src/quest/fixtures/quest-contract.json``. The shell's vitest
(``src/quest/cards/CardFor.test.tsx``) reads it and fails on any pair the card dispatcher has no
renderer for.

The fixture also carries what the shell's other tests and the mock server read from the engine
itself, so that none of them retypes it: the seven stages and their names, each Router question's
and declaration's place and name, each sweep size's heading and action, two quest logs of the
NHANES fixture under Predict (Models open; Fit pressed and Results reached) with what "Confirm all"
records in each stage, the Fit lock as the engine words it (held, ready, pressed, unlocked under
Estimate and with no goal), the floor a gain over the no-predictor baseline must reach, and the
decision kinds each question that can carry the ledger's ask card records.

When the engine emits a pair the fixture does not hold (or stops emitting one), this test writes
the fixture anew and fails, so the change reaches the vitest in the same commit.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from turbotab.core import ask, fit_press, quest, sweep
from turbotab.core.decisions import SLOTS, ProjectState
from turbotab.core.interview import route
from turbotab.core.models import baseline
from turbotab.core.sequence import question_of
from turbotab.core.tests import path_fuzzer
from turbotab.core.voice import question_name

FIXTURE = (Path(__file__).resolve().parents[2] / "frontend" / "src" / "quest" / "fixtures"
           / "quest-contract.json")
# Every (source, key) of 300 journeys appears by seed 4, and every one in every dataset and goal it
# appears in by seed 17 (measured on 2026-10-17); 40 seeds keep that margin.
SEEDS = 40
GOALS = ("none", "inference", "prediction")
# A fit held for the press, as long as the shelf can estimate one (FOUNDATION §7, "about 30 min").
HELD_SECONDS = 1800.0


def _goal(state: Any) -> str:
    return str(state.purpose) if state.purpose is not None else "none"


@lru_cache(maxsize=1)
def pairs() -> list[dict[str, Any]]:
    """Each quest line's ``(source, key)`` over the seeded journeys, where it was seen."""
    seen: dict[tuple[str, str], dict[str, set[str]]] = {}
    for seed in range(SEEDS):
        journey = path_fuzzer.run_journey(seed)
        for snap in journey.snapshots:
            for stage in snap.log.stages:
                for line in stage.lines:
                    at = seen.setdefault((line.source, line.key),
                                         {"stages": set(), "labels": set(), "seen": set()})
                    at["stages"].add(stage.key)
                    at["labels"].add(line.label)
                    at["seen"].add(f"{journey.fixture}/{_goal(snap.state)}")
    return [{"source": source, "key": key, **{name: sorted(v) for name, v in at.items()}}
            for (source, key), at in sorted(seen.items())]


def _log(stop: str, pressed: bool) -> dict[str, Any]:
    """The NHANES fixture under Predict, every Router question before ``stop`` answered by its
    first option, the fit fresh, Fit pressed or not: its quest log as GET /quest serves it, and
    what "Confirm all" records for each stage with a sweep (``sweep.sweep_lines``)."""
    fx = path_fuzzer.NHANES
    state, artifacts, records = path_fuzzer.answered_through(fx, "prediction", stop=stop)
    stages = {"fit": {"status": "fresh"}}
    steps = route(state, stages, artifacts, records, pressed=pressed)
    log = path_fuzzer.quest_of(state, records, steps, stages, artifacts, fx, pressed, pressed)
    swept = {s.key: [l.model_dump(mode="json") for l in sweep.sweep_lines(s)]
             for s in log.stages if s.sweep is not None}
    return {"log": log.model_dump(mode="json"), "swept": swept}


def _locks() -> dict[str, Any]:
    state, _artifacts, records = path_fuzzer.answered_through(path_fuzzer.NHANES, "prediction")
    held = fit_press.fit_lock(state, records, pressed=False, held=True, estimate=HELD_SECONDS)
    pressed = fit_press.fit_lock(state, records, pressed=True, held=False, estimate=HELD_SECONDS)
    ready = fit_press.fit_lock(state, records, pressed=False, held=False, estimate=None)
    state, _artifacts, records = path_fuzzer.answered_through(path_fuzzer.NHANES, "inference")
    unlocked = fit_press.fit_lock(state, records, pressed=False, held=False, estimate=None)
    no_goal = fit_press.fit_lock(ProjectState(), [], pressed=False, held=False, estimate=None)
    return {"estimate_seconds": HELD_SECONDS, "held": held.model_dump(mode="json"),
            "pressed": pressed.model_dump(mode="json"), "ready": ready.model_dump(mode="json"),
            "unlocked": unlocked.model_dump(mode="json"), "no_goal": no_goal.model_dump(mode="json")}


@lru_cache(maxsize=1)
def contract() -> dict[str, Any]:
    return {
        "written_by": "turbotab/core/tests/test_quest_renderer_contract.py",
        "seeds": SEEDS,
        "datasets": [fx.name for fx in path_fuzzer.FIXTURES],
        "goals": list(GOALS),
        "pairs": pairs(),
        "stages": [{"key": key, "name": name} for key, name in quest.STAGES],
        "questions": {key: {"stage": p.stage, "label": p.label, "order": p.order,
                            "name": question_name(key)} for key, p in quest.QUESTIONS.items()},
        "declarations": {d.kind: {"stage": d.place.stage, "label": d.place.label,
                                  "order": d.place.order, "name": d.name, "counted": d.counted}
                         for d in quest.DECLARATIONS},
        "sweep_words": {str(n): list(sweep._sweep_words(n)) for n in range(1, 9)},
        "logs": {"models_open": _log("models", pressed=False),
                 "results_reached": _log("open_seal", pressed=True)},
        "fit_locks": _locks(),
        # The floor a score's gain over the no-predictor baseline must reach (the mock's verdict).
        "min_gain": baseline.MIN_GAIN,
        # The decision kinds each question that can carry the ledger's ask card records (the
        # shell's record/besideAsk.ts `ASKED_BY`: where a refusal's readings are the card's to say).
        "asked_by": {q: sorted(k for k in SLOTS if question_of(k) == q)
                     for q in sorted(ask.CONSUMERS)},
    }


def test_every_dataset_and_goal_is_walked():
    walked = {s for p in pairs() for s in p["seen"]}
    expected = {f"{fx.name}/{goal}" for fx in path_fuzzer.FIXTURES for goal in GOALS}
    assert expected <= walked, f"not walked: {sorted(expected - walked)}"


def test_the_shell_fixture_holds_every_pair_the_engine_emits():
    now = contract()
    held = json.loads(FIXTURE.read_text(encoding="utf-8")) if FIXTURE.exists() else None
    if held != now:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(now, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    was = {(p["source"], p["key"]) for p in (held or {}).get("pairs", [])}
    gained = sorted({(p["source"], p["key"]) for p in now["pairs"]} - was)
    assert held == now, (f"{FIXTURE.name} was out of date and is written anew (pairs gained: "
                         f"{gained or 'none'}); commit it, then run the shell's vitest.")


def test_the_samples_are_the_states_they_say():
    logs = json.loads(json.dumps(contract()["logs"]))
    early, late = logs["models_open"]["log"], logs["results_reached"]["log"]
    models = next(s for s in early["stages"] if s["key"] == "models")
    assert models["reached"] and not models["progress"]["complete"]
    assert any(l["status"] == "open" for l in models["lines"])
    assert early["fit"]["pressed"] is False
    results = next(s for s in late["stages"] if s["key"] == "results")
    assert results["reached"] and late["fit"]["pressed"] is True
    # The sweeps the shell's "Confirm all" is held to: Models' three defaults under Predict.
    assert [l["key"] for l in logs["results_reached"]["swept"]["models"]] == [
        "form", "set_validation", "set_levers"]
