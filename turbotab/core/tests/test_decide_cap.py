"""b1 (SURFACING_POLICY §4.2 and SWARM_PLAN §11, the Decide cap per stage; the vision audit's V6): how
many Decide lines each reference journey answers in each stage, counted from its cached capture
as captured, with no fit, and a cap per stage that no capture may exceed.

What this measures, and what it cannot: the cached captures were taken on the engine of 2026-10-05
(commit 2ba3c5727a), and each is counted as captured: an ask lists the readings its refusal or
card listed then, and nothing is read again from the data. So the caps below are that engine's
counts, not today's Decide load: a count here moves with a recapture, or with a change to where
``quest`` places a kind of answer, and an engine change that asks more shows only in a recapture.
A later engine's load is measured on one journey: the vision audit's walk of dietary-inference on
60986882, an ancestor of this package's base (``WALKED_LOAD``), above the captured caps in Your
data and Models. A cap measured on today's engine needs a recapture at a package's head, which the
owner schedules.

A capture (``reference.journeys``) records every answer a journey posted, where each came from,
and every refusal it met. Its Decide lines, stage by stage (``quest``'s places):

* each Router question or declaration the journey answered whose place is a Decide, once however
  many posts answered it (the adjustment card's groups are one line), each finding's disposition
  its own line, and never a Confirm or For the record answer (``confirm_sweep``, the split under
  Estimate);
* each consumer's ask: a refusal that asks only for readings (``journeys.ASKING``), in the stage of
  the decision it refused, its answers part of it; and an ask card's readings answered before the
  question whose card asked them, in that question's stage;
* each press on a role line (a role, or one judgment's group of roles, confirmed after the roles
  answer), in Your data.

The readings each ask lists are counted too, as the capture lists them: the columns a refusal's
ways forward name, or those a card's answers settle (a block of more than four names its first
four and counts the rest).

    python -m pytest turbotab/core/tests/test_decide_cap.py -s   # prints each journey's counts
"""
from __future__ import annotations

import gzip
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

from turbotab.core import quest
from turbotab.core.reference import journeys as J

STAGES = ("data", "question", "whos_in", "first_look", "models", "results", "writeup")

# The 2026-10-05 engine's counts, not today's caps: the Decide lines each stage held at most in
# the reference journeys' cached captures (taken on commit 2ba3c5727a), counted as captured. A
# capture that answers more in a stage fails; when every capture's count in a stage falls, the cap
# falls with it (``test_the_captured_cap_is_the_most_any_capture_answers``). A recapture is
# counted the same way, and a stage it raises above its cap is a Decide load to answer for, not a
# number to retype.
CAPTURED_DECIDE_CAP: dict[str, int] = {"data": 6, "question": 4, "whos_in": 7, "models": 9,
                                       "results": 1}
# The readings the asks of one stage list at most, by the same rule: an ask is one line, and a
# reading added to an ask already there is a question more that the line count does not see.
CAPTURED_ASKED_CAP: dict[str, int] = {"whos_in": 3, "models": 31}
# The Decide load on dietary-inference at 60986882, (lines, readings asked) per stage: the vision
# audit's walk (2026-11-04), counted from its capture (``WALK``) the same way, as captured. That
# engine presses nine role lines in Your data (two blocks and seven single roles, where the
# 2026-10-05 capture pressed one block), and in Models asks age's code-or-amount reading on the
# form question's card (the 2026-10-05 capture asked it within the fit's ask) and sugar's nesting
# in carb at the substitution after its unit: Your data and Models above the captured caps.
WALKED_LOAD: dict[str, tuple[dict[str, int], dict[str, int]]] = {
    "dietary-inference": ({"data": 12, "question": 2, "whos_in": 4, "models": 11},
                          {"whos_in": 3, "models": 5})}
WALK = (Path(__file__).resolve().parent / "fixtures" / "captures"
        / "dietary-inference_60986882.json.gz")

NAMED = re.compile(r"^`(?P<column>[^`]+)`")
ITEM = re.compile(r"(?P<reading>[a-z_]+):(?P<column>[^=,]+)=")


def _place(kind: str, purpose: str | None) -> quest.Place | None:
    from turbotab.core.sequence import question_of

    if kind == quest.FOLLOWS_ITS_STAGE or kind == "revert":
        return None  # a Confirm sweep, or a record undone: no line of its own
    key = question_of(kind)
    if key is not None and (key, purpose) in quest.GOAL_PLACES:
        return quest.GOAL_PLACES[(key, purpose)]
    return quest.kind_place(kind)


def _summary(answer: dict[str, Any]) -> dict[str, Any]:
    try:
        body = json.loads(answer.get("summary") or "")
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _readings_answered(answer: dict[str, Any]) -> tuple[list[tuple[str, str]], int]:
    """The (reading, column) pairs a reading answer shows, and how many it settles in all (a block
    of more than four lists its first four: ``journeys.summarize``)."""
    if answer["kind"] == "confirm_readings":
        text = str(answer.get("summary") or "")
        n = int(re.match(r"\s*([\d,]+)", text).group(1).replace(",", "")) if re.match(
            r"\s*[\d,]+", text) else 0
        listed = text.split("): ", 1)[1] if "): " in text else ""
        return [(m["reading"], m["column"].strip()) for m in ITEM.finditer(listed)], n
    body = _summary(answer)
    if answer["kind"] == "confirm_reading":
        return [(str(body.get("reading")), str(body.get("column")))], 1
    if answer["kind"] == "set_column_unit":
        return [("unit", str(body.get("column")))], 1
    return [], 0


def _ask_from_exits(labels: list[str]) -> dict[str, Any]:
    """A refusal's readings, by the columns its ways forward name."""
    return {"columns": list(dict.fromkeys(m["column"] for m in map(NAMED.match, labels) if m)),
            "hidden": 0}


def _ask_from_answer(pairs: list[tuple[str, str]], n: int) -> dict[str, Any]:
    return {"columns": [c for _, c in pairs], "hidden": max(0, n - len(pairs))}


def decide_lines(capture: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each stage's Decide lines in ``capture``, as captured, by key: an answered line (``None``),
    or an ask with the readings it lists (``columns``; ``hidden`` those a long block's summary
    leaves unnamed)."""
    purpose = capture["journey"].get("purpose")
    out: dict[str, dict[str, Any]] = {}
    pending: tuple[str, str] | None = None  # the ask whose answers follow
    card: list[dict[str, Any]] = []  # an ask card's readings, before the question that asked them

    def ask(stage: str, kind: str, found: dict[str, Any]) -> tuple[str, str]:
        n = sum(1 for k in out.get(stage, {}) if k.startswith("ask:"))
        key = f"ask:{kind}:{n}"
        out.setdefault(stage, {})[key] = found
        return stage, key

    for answer in capture["answers"]:
        kind = str(answer["kind"])
        refusal = answer.get("refusal")
        if refusal:
            if refusal.get("code") in J.ASKING:
                place = _place(kind, purpose)
                pending = ask(place.stage, kind, _ask_from_exits(list(refusal.get("exits") or [])))
            continue  # another refusal re-asks its own line, whose answer follows
        if kind in J.READING_KINDS:
            pairs, n = _readings_answered(answer)
            if pending is not None:
                continue  # the ask's answer
            if any(r == "role" for r, _ in pairs):
                place = quest.kind_place(kind)
                out.setdefault(place.stage, {})[f"{place.item}:{answer.get('seq')}"] = None
            else:
                card.append(_ask_from_answer(pairs, n))
            continue
        pending = None
        place = _place(kind, purpose)
        if place is None or place.label != quest.DECIDE:
            continue
        if card:
            found = {"columns": [c for a in card for c in a["columns"]],
                     "hidden": sum(a["hidden"] for a in card)}
            ask(place.stage, f"card:{kind}", found)
            card = []
        body = _summary(answer)
        key = (f"{place.item}:{body.get('finding_id')}"
               if kind in ("apply_repair", "defer_finding", "dismiss_finding") else place.item)
        out.setdefault(place.stage, {})[key] = None
    return out


def counts(lines: dict[str, dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    """Each stage's Decide lines, and the readings its asks list."""
    asked = {s: sum(len(v["columns"]) + v["hidden"] for v in ls.values() if v is not None)
             for s, ls in lines.items()}
    return ({s: len(ls) for s, ls in lines.items()}, {s: n for s, n in asked.items() if n})


@lru_cache(maxsize=None)
def _capture(name: str) -> dict[str, Any]:
    found = J.load_capture(name)
    assert found is not None, name
    return found


def as_captured(name: str) -> tuple[dict[str, int], dict[str, int]]:
    """``name``'s Decide lines and asked readings per stage, counted from its capture as captured."""
    return counts(decide_lines(_capture(name)))


def _table() -> str:
    rows = []
    for name in J.JOURNEYS:
        lines, asked = as_captured(name)
        rows.append(f"{name:24} " + "  ".join(
            f"{s} {lines[s]} (asked {asked.get(s, 0)})" for s in STAGES if lines.get(s)))
    return "\n".join(rows)


# ── the cap ──────────────────────────────────────────────────────────────────


def over_cap(name: str, lines: dict[str, int], asked: dict[str, int]) -> list[str]:
    """Each stage where ``name`` answers more Decide lines, or its asks list more readings, than
    the captured cap allows."""
    return ([f"{name} {s}: {n} Decide lines > {CAPTURED_DECIDE_CAP.get(s, 0)}"
             for s, n in lines.items() if n > CAPTURED_DECIDE_CAP.get(s, 0)]
            + [f"{name} {s}: {n} readings asked > {CAPTURED_ASKED_CAP.get(s, 0)}"
               for s, n in asked.items() if n > CAPTURED_ASKED_CAP.get(s, 0)])


def test_no_cached_capture_answers_more_than_its_captured_cap():
    """The guard stays silent: each 2026-10-05 capture, counted as captured, answers no more
    Decide lines in a stage, and its asks list no more readings, than the captured cap."""
    print("\nDecide lines per stage, as captured:\n" + _table())
    over = [o for name in J.JOURNEYS for o in over_cap(name, *as_captured(name))]
    assert not over, "\n".join(over) + "\n" + _table()


def test_the_audit_walk_measures_dietary_inference_load_on_60986882():
    """The vision audit's walk of dietary-inference on 60986882, counted as captured: the record of
    that engine's load, above the 2026-10-05 caps in Your data and Models (a recapture at a
    package's head, owner-scheduled, re-measures every journey's)."""
    walk = json.loads(gzip.decompress(WALK.read_bytes()))
    assert walk["journey"]["name"] == "dietary-inference"
    assert walk["captured"]["commit"] == "60986882c0"
    name = "dietary-inference"
    lines = decide_lines(walk)
    got = counts(lines)
    assert got == WALKED_LOAD[name], got
    roles = [k for k in lines["data"] if k.startswith("decision:readings-ask-card:")]
    assert len(roles) == 9  # each role press its own line
    assert over_cap(name, *got) == [
        f"{name} data: {got[0]['data']} Decide lines > {CAPTURED_DECIDE_CAP['data']}",
        f"{name} models: {got[0]['models']} Decide lines > {CAPTURED_DECIDE_CAP['models']}"]


def test_the_cap_catches_one_line_more_and_one_reading_more():
    """The guard fires: the journey with the most Models lines, given one Decide more there (a
    declaration it never answered), or one reading more on its fit's ask, is over the cap."""
    name = "dietary-inference"
    capture = _capture(name)
    lines, asked = as_captured(name)
    assert lines["models"] == CAPTURED_DECIDE_CAP["models"] and not over_cap(name, lines, asked)
    more = {**capture, "answers": [*capture["answers"],
                                   _answer("set_sensitivity", {"evalue": True})]}
    got = counts(decide_lines(more))
    assert got[0]["models"] == CAPTURED_DECIDE_CAP["models"] + 1
    assert over_cap(name, *got) == [f"{name} models: {CAPTURED_DECIDE_CAP['models'] + 1} Decide "
                                    f"lines > {CAPTURED_DECIDE_CAP['models']}"]
    wider = [dict(a) for a in capture["answers"]]
    ask = next(a for a in wider if a["kind"] == "select_models" and a.get("refusal"))
    ask["refusal"] = {**ask["refusal"], "exits": [
        *ask["refusal"]["exits"], *(f"`x{i}` holds codes for categories (one indicator per level)"
                                    for i in range(CAPTURED_ASKED_CAP["models"]))]}
    got = counts(decide_lines({**capture, "answers": wider}))
    assert got[0] == lines and got[1]["models"] > CAPTURED_ASKED_CAP["models"]
    assert over_cap(name, *got) == [f"{name} models: {got[1]['models']} readings asked > "
                                    f"{CAPTURED_ASKED_CAP['models']}"]


def test_the_captured_cap_is_the_most_any_capture_answers():
    """So a count can only fall: each cap is the most any cached capture answers in its stage,
    counted as captured."""
    most: dict[str, int] = {}
    most_asked: dict[str, int] = {}
    for name in J.JOURNEYS:
        lines, asked = as_captured(name)
        for s, n in lines.items():
            most[s] = max(most.get(s, 0), n)
        for s, n in asked.items():
            most_asked[s] = max(most_asked.get(s, 0), n)
    assert most == CAPTURED_DECIDE_CAP, _table()
    assert most_asked == CAPTURED_ASKED_CAP, _table()


# ── the counter itself, on captures built here ───────────────────────────────


def _answer(kind: str, body: Any = None, **extra: Any) -> dict[str, Any]:
    summary = body if isinstance(body, str) else json.dumps(body or {})
    return {"kind": kind, "status": 200, "summary": summary, **extra}


def _refused(kind: str, code: str, exits: list[str]) -> dict[str, Any]:
    return {"kind": kind, "status": 409, "summary": "{}",
            "refusal": {"code": code, "message": "", "exits": exits}}


CODE_EXITS = ["Confirm each of the 2 as shown",
              "`kcal` is a count or an amount (a slope; a unit's rows averaged)",
              "`kcal` holds codes for categories (one indicator per level; a unit's rows take "
              "their most frequent value)",
              "`cycle` holds codes for categories (one indicator per level; a unit's rows take "
              "their most frequent value)",
              "`cycle` is a count or an amount (a slope; a unit's rows averaged)"]


def _constructed() -> dict[str, Any]:
    return {"journey": {"purpose": "inference"}, "answers": [
        _answer("set_lens", {"lenses": ["dietary"]}),
        _answer("apply_repair", {"finding_id": "a"}), _answer("apply_repair", {"finding_id": "b"}),
        _answer("set_target", {"column": "glucose_mg_dl"}),
        _answer("set_outcome_unit", {"column": "glucose_mg_dl", "unit": "mg/dL"}),
        _answer("set_outcome_unit", {"column": "LBXGLU", "unit": "mg/dL"}),
        _answer("set_purpose", {"purpose": "inference"}),
        _answer("set_roles", "covariate: age"),
        _answer("confirm_reading", {"reading": "role", "column": "age", "value": "covariate"},
                seq=9),
        _answer("confirm_readings", "2 readings (role 2): role:a=flag, role:b=flag", seq=10),
        _answer("set_split", {"holdout": 0.0}),  # For the record under Estimate
        _answer("set_adjustment", {"exposure": "x"}), _answer("set_adjustment", {"exposure": "x"}),
        _answer("confirm_readings", "1 reading (code_or_count 1): code_or_count:age=amount"),
        _answer("set_forms", {"forms": {}}),
        _refused("select_models", "reading_unsettled", CODE_EXITS),
        _answer("confirm_readings", "2 readings (code_or_count 2): code_or_count:kcal=amount, "
                                    "code_or_count:cycle=code"),
        _answer("select_models", {"models": ["linear"]}),
        _refused("set_substitution", "reading_unsettled", ["`kcal` is a count or an amount (a "
                                                           "slope; a unit's rows averaged)"]),
        _answer("confirm_readings", "1 reading (code_or_count 1): code_or_count:kcal=amount"),
        _answer("set_substitution", {"donor": "x"}),
        _answer("confirm_sweep", {"stage": "models"}),
    ]}


def test_the_counter_counts_each_decide_line_once_in_its_stage():
    lines = decide_lines(_constructed())
    assert set(lines["data"]) == {"q:lens", "decision:finding-disposition:a",
                                  "decision:finding-disposition:b", "q:roles",
                                  "decision:readings-ask-card:9", "decision:readings-ask-card:10"}
    assert set(lines["question"]) == {"q:target", "q:goal", "q:outcome_unit"}
    assert "whos_in" not in lines  # the split under Estimate is For the record
    models = lines["models"]
    assert set(models) == {"q:adjustment", "ask:card:set_forms:0", "q:form",
                           "ask:select_models:1", "q:models", "ask:set_substitution:2",
                           "decision:substitution-pair"}
    assert models["ask:card:set_forms:0"]["columns"] == ["age"]
    assert models["ask:select_models:1"]["columns"] == ["kcal", "cycle"]
    assert models["ask:set_substitution:2"]["columns"] == ["kcal"]
    assert counts(lines)[1] == {"models": 4}


@pytest.mark.parametrize("name", list(J.JOURNEYS))
def test_every_reference_journey_has_a_capture_to_count(name):
    capture = _capture(name)
    assert capture["journey"]["name"] == name and capture["answers"]
    assert all(a["kind"] for a in capture["answers"])
