"""W1-F · the Predict declaration cards post what the server's models accept (W1_PART2_PLAN §2, W1-F).

The frontend cards (turbotab/frontend/src/quest/cards/predict) post six decisions: the intended use,
the selection, the validation scheme, the levers, the updating and the explanations. The vitest
(cards.test.tsx) presses each option and asserts the card posts exactly the payload written in
``payloads.fixture.json``; this file holds that fixture to the models in ``core/decisions.py``.

The expectations come from the models' own fields and constraints (``typing.get_args`` of each
Literal, pydantic's ``Ge``/``Le``/``Gt``/``Lt`` metadata, each field's default), not from the cards
or from a stored engine output, and a press the card must not send is shown to be one the model
refuses:

* every payload a card posts is accepted by ``parse_decision`` and comes back whole;
* each choice lists exactly the values the model accepts, and each value has a card option that
  posts it; an option left untouched posts the model's defaults for what it does not ask;
* the bounds the card checks are the models' bounds, and a typed value outside them is one the
  model refuses (the guard fires), while every filled-in value inside them is posted and accepted
  (it stays silent, the first test above);
* the seal's re-answer keeps a validation scheme: ``set_split`` replaces the whole split, so the
  card must send the scheme back with the draw (the premise, shown on the engine's fold).
"""
from __future__ import annotations

import json
import typing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from turbotab.core import decisions as d
from turbotab.core import models  # noqa: F401  (registers the selection and explain validators)
from turbotab.core.decisions import DecisionRecord, ProjectState, parse_decision

FIXTURE = Path(__file__).resolve().parents[2] / "frontend/src/quest/cards/predict/payloads.fixture.json"
DOC = json.loads(FIXTURE.read_text("utf-8"))
CASES = DOC["cases"]
MODELS: dict[str, type] = {
    "set_intended_use": d.SetIntendedUse,
    "set_selection": d.SetSelection,
    "set_validation": d.SetValidation,
    "set_levers": d.SetLevers,
    "set_updating": d.SetUpdating,
    "set_explain": d.SetExplain,
}
T0 = datetime(2026, 10, 17, tzinfo=timezone.utc)


def label(case: dict[str, Any]) -> str:
    fill = ",".join(f"{k}={v}" for k, v in (case.get("fill") or {}).items())
    toggle = ",".join(case.get("toggle") or [])
    return f"{case['card']}|{case['press']}|{fill}|{toggle}|{case.get('note', '')}"


def literal_values(model: type, field: str) -> list[Any]:
    """The values a model's field accepts (None, where it is optional, left out)."""
    annotation = model.model_fields[field].annotation
    found: list[Any] = []

    def walk(a: Any) -> None:
        if typing.get_origin(a) is typing.Literal:
            found.extend(typing.get_args(a))
        else:
            for part in typing.get_args(a):
                if part is not type(None):
                    walk(part)

    walk(annotation)
    return found


def accepts_none(model: type, field: str) -> bool:
    return type(None) in typing.get_args(model.model_fields[field].annotation)


def constraints(model: type, field: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for meta in model.model_fields[field].metadata:
        for name in ("ge", "le", "gt", "lt"):
            if hasattr(meta, name):
                out[name] = getattr(meta, name)
    return out


def defaults_of(model: type) -> dict[str, Any]:
    """The model's own default for every field that has one, ``kind`` apart."""
    out: dict[str, Any] = {}
    for name, field in model.model_fields.items():
        if name == "kind" or field.is_required():
            continue
        out[name] = field.get_default(call_default_factory=True)
    return out


def record(seq: int, decision: dict[str, Any]) -> DecisionRecord:
    return DecisionRecord(id=f"r{seq}", seq=seq, at=T0, decision=parse_decision(decision))


# ── every payload is the server's ────────────────────────────────────────────

POSTED = [c for c in CASES if c.get("payload") is not None]
BLOCKED = [c for c in CASES if c.get("payload") is None]


@pytest.mark.parametrize("case", POSTED, ids=[label(c) for c in POSTED])
def test_a_payload_a_card_posts_is_accepted_and_comes_back_whole(case):
    payload = case["payload"]
    parsed = parse_decision(payload)
    assert type(parsed) is MODELS[case["card"]]
    # Nothing the card sends is dropped or filled in by the model: it is the whole of the model.
    assert parsed.model_dump(mode="json") == payload


def test_six_cards_and_no_other_post_decisions():
    assert {c["card"] for c in CASES} == set(MODELS)
    assert set(DOC["choices"]) == set(MODELS)


# ── the options are exactly the values the model accepts ─────────────────────


@pytest.mark.parametrize("card", sorted(MODELS))
def test_each_choice_lists_exactly_the_values_the_model_accepts(card):
    for field, offered in DOC["choices"][card].items():
        accepted = literal_values(MODELS[card], field)
        assert len(offered) == len(set(offered)), (card, field)
        assert sorted(offered) == sorted(accepted), (card, field)


@pytest.mark.parametrize("card", sorted(DOC["selects"]))
def test_each_select_lists_the_values_the_model_accepts_and_a_blank_only_where_it_may_be_none(card):
    for field, select in DOC["selects"][card].items():
        accepted = literal_values(MODELS[card], field)
        assert sorted(select["values"]) == sorted(accepted), (card, field)
        assert select["blank"] == accepts_none(MODELS[card], field), (card, field)


@pytest.mark.parametrize("card", sorted(MODELS))
def test_every_option_has_a_case_that_posts_its_value(card):
    for field, offered in DOC["choices"][card].items():
        for value in offered:
            press = f"{field}:{value}"
            posted = [c for c in POSTED if c["card"] == card and c["press"] == press]
            assert posted, f"no case presses {press}"
            assert all(c["payload"][field] == value for c in posted), press
    # and no case presses an option the card does not list
    listed = {f"{f}:{v}" for f, vs in DOC["choices"][card].items() for v in vs}
    assert {c["press"] for c in CASES if c["card"] == card} <= listed


def test_an_option_left_untouched_posts_the_models_defaults_for_what_it_does_not_ask():
    untouched = [c for c in POSTED
                 if not (c.get("fill") or c.get("toggle") or c.get("recorded"))]
    assert untouched
    for case in untouched:
        model = MODELS[case["card"]]
        field, value = case["press"].split(":", 1)
        expected = {"kind": case["card"], **defaults_of(model), field: value}
        assert case["payload"] == expected, label(case)


def test_the_defaults_in_the_fixture_are_the_models():
    for card, model in MODELS.items():
        assert DOC["defaults"][card] == defaults_of(model), card


def test_what_a_card_never_asks_it_sends_as_the_model_defaults_it():
    for card, fixed in DOC["fixed"].items():
        for field, value in fixed.items():
            assert defaults_of(MODELS[card])[field] == value, (card, field)
            assert all(c["payload"][field] == value for c in POSTED if c["card"] == card)


def test_the_values_the_cards_never_send_are_refused_by_the_engine_under_prediction():
    """The cards offer a subset of what the models accept, and the engine is why: selection
    outside the resampling gives a false performance number, and an explanation is never an
    effect. Each is refused, with its reason, where the card would otherwise have to offer it."""
    state = ProjectState(target="LBXGLU", task="regression", purpose="prediction")
    ctx = {"state": state, "records": lambda: []}
    with pytest.raises(d.Refusal) as outside:
        d.validate({"kind": "set_selection", "method": "elastic_net", "where": "outside"}, ctx)
    assert outside.value.code == "selection_outside_resampling"
    with pytest.raises(d.Refusal) as effect:
        d.validate({"kind": "set_explain", "as_effect": True}, ctx)
    assert effect.value.code == "explanation_is_not_an_effect"


# ── the bounds ───────────────────────────────────────────────────────────────


def test_the_bounds_in_the_fixture_are_the_models_and_cover_every_bounded_field():
    for card, model in MODELS.items():
        bounded = {f for f in model.model_fields if constraints(model, f)}
        assert set(DOC["bounds"].get(card, {})) == bounded, card
        for field in bounded:
            assert DOC["bounds"][card][field] == constraints(model, field), (card, field)


INTEGER_FIELDS = ("folds", "repeats", "n_boot", "keep", "q", "reseeds")


def with_field(case: dict[str, Any], testid: str, text: str) -> dict[str, Any]:
    """The payload a card would send if it sent ``text`` typed into the field ``testid``."""
    model = MODELS[case["card"]]
    field, value = case["press"].split(":", 1)
    payload = {"kind": case["card"], **defaults_of(model), field: value}
    name = testid.removeprefix("field-")
    assert name in model.model_fields, testid
    payload[name] = int(text) if name in INTEGER_FIELDS else float(text)
    return payload


@pytest.mark.parametrize("case", [c for c in BLOCKED if c.get("fill")],
                         ids=[label(c) for c in BLOCKED if c.get("fill")])
def test_a_value_the_card_will_not_send_is_one_the_model_refuses(case):
    [(testid, text)] = case["fill"].items()
    with pytest.raises(ValidationError):
        parse_decision(with_field(case, testid, text))


def test_a_scheme_by_the_groups_of_a_column_needs_the_column_and_the_card_waits_for_it():
    [case] = [c for c in BLOCKED if not c.get("fill")]
    assert case["press"] == "validation:internal_external"
    with pytest.raises(ValidationError):
        parse_decision({"kind": "set_validation", "validation": "internal_external"})
    # ...and a column named by any other scheme is refused too, so the card sends it with
    # internal-external validation alone.
    with pytest.raises(ValidationError):
        parse_decision({"kind": "set_validation", "validation": "kfold",
                        "cluster": "cycle_begin_year"})


# ── the seal keeps the scheme ────────────────────────────────────────────────

SCHEME = ("folds", "validation", "repeats", "n_boot", "cluster", "nested_cv")


def scheme_of(split: Any) -> dict[str, Any]:
    dump = split if isinstance(split, dict) else split.model_dump()
    return {k: dump[k] for k in SCHEME}


def replay(*after_the_draw: dict[str, Any]) -> d.ProjectState:
    base = [record(1, {"kind": "set_target", "column": "LBXGLU"}),
            record(2, {"kind": "set_purpose", "purpose": "prediction"}),
            record(3, {"kind": "set_split", "holdout": 0.2, "seed": 0})]
    return d.fold([*base, *[record(4 + i, x) for i, x in enumerate(after_the_draw)]])


def test_the_premise_set_split_replaces_the_scheme_set_with_set_validation():
    """Why the seal's card sends the scheme back: the engine's fold writes the whole split."""
    changed = {"kind": "set_validation", "folds": 10, "validation": "repeated_kfold",
               "repeats": 20, "n_boot": 700, "nested_cv": True}
    kept = replay(changed)
    assert scheme_of(kept.split)["validation"] == "repeated_kfold"
    reset = replay(changed, {"kind": "set_split", "holdout": 0.1, "seed": 0})
    assert scheme_of(reset.split) == scheme_of(d.SplitSpec(holdout=0.1))


@pytest.mark.parametrize("seal", DOC["seal"], ids=[s["name"] for s in DOC["seal"]])
def test_the_seals_card_keeps_the_scheme_and_sends_the_draw(seal):
    payload = seal["payload"]
    parsed = parse_decision(payload)
    assert isinstance(parsed, d.SetSplit)
    assert parsed.model_dump(mode="json") == payload
    current = seal["current"]
    if current is None:
        # No split recorded: there is no scheme to keep, and the model's defaults stand.
        assert scheme_of(payload) == scheme_of(d.SplitSpec(holdout=payload["holdout"]))
        after = replay(payload)
    else:
        after = replay({"kind": "set_validation", **scheme_of(current)}, payload)
        assert scheme_of(after.split) == scheme_of(current)
    assert after.split.holdout == payload["holdout"] and after.split.seed == payload["seed"]
