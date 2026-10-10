"""P0.12 · what the server serves on a card speaks plainly.

A dietary journey through the real server, from the opening questions to the adjustment set: every
card the person reads on the way (the interview's reasons and questions, the proposals' estimand,
adjustment and form cards) and every refusal the server answers with is walked, and no sentence
in it uses "exposure", "confounder" or "estimand". The machine's own fields (a decision's keys, a
column's role, a source) are quiet labels and identifiers, not card words, and are skipped.

The static scans of ``tests/test_plain_words.py`` cover the branches this journey does not reach.
"""
from __future__ import annotations

import time
from typing import Any

import pytest

from turbotab.core import plain, teaching, voice
from turbotab.core.tests.acceptance.server_drive import local_server, open_project
from turbotab.core.tests.acceptance.test_estimand_chain import (
    DIET_ROLES, DIET_TRUTH, _diet, _open, _seal)
from turbotab.core.tests.truths import Truth, answer_adjustment


ANSWER = {"sex": {"causes_exposure": "yes", "causes_outcome": "yes", "after_exposure": "no"}}


def spoken(node: Any, path: str = "") -> list[tuple[str, str]]:
    """Every sentence-length string in a served payload that a card can carry, with its path."""
    found: list[tuple[str, str]] = []
    if isinstance(node, str):
        if len(node.split()) >= 3:
            found.append((path, node))
    elif isinstance(node, dict):
        for key, value in node.items():
            if key not in plain.QUIET_KEYS:
                found += spoken(value, f"{path}.{key}")
    elif isinstance(node, list):
        for value in node:
            found += spoken(value, f"{path}[]")
    return found


def impolite(node: Any) -> list[tuple[str, str]]:
    return [(p, s[:100]) for p, s in spoken(node) if plain.forbidden_terms(s)]


@pytest.fixture(scope="module")
def served(tmp_path_factory) -> dict[str, Any]:
    folder = tmp_path_factory.mktemp("plain")
    csv = folder / "diet.csv"
    _diet(600).to_csv(csv, index=False)
    truth = Truth(DIET_TRUTH, fixture="the dietary chain")
    seen: dict[str, Any] = {}
    with local_server(folder / "home") as client:
        drive = open_project(client, csv, truth)
        _open(drive, "dietary", "dm", "True")
        _seal(drive, DIET_ROLES)
        drive.reach("estimand")
        seen["interview"] = drive.view()["interview"]
        seen["estimand"] = drive.artifact("proposals")
        # refusals before the study factor is declared: a column the data does not have, a set
        # of study factors with only one in the model, and an adjustment asked too early
        for name, body in {
                "unknown_column": {"kind": "set_estimand", "exposure": "no_such_column",
                                   "effect": "total", "measure": "risk_difference"},
                "family_of_one": {"kind": "set_estimand", "family": True, "effect": "total",
                                  "measure": "exposure_mean_difference"},
                "no_estimand_yet": {"kind": "set_adjustment", "exposure": "protein_g",
                                    "answers": ANSWER}}.items():
            seen[f"refused:{name}"] = drive.post(body).json()
        drive.decide({"kind": "set_estimand", "exposure": "protein_g", "effect": "total",
                      "contrast": "substitution", "measure": "risk_difference"})
        drive.reach("adjustment")
        seen["adjustment"] = drive.artifact("proposals")
        seen["refused:other_exposure"] = drive.post(
            {"kind": "set_adjustment", "exposure": "age", "answers": ANSWER}).json()
        answer_adjustment(drive.post, seen["adjustment"]["adjustment"], truth)
        end = time.monotonic() + 120
        while time.monotonic() < end:
            seen["after"] = drive.artifact("proposals")
            if seen["after"].get("model_sequence") is not None:
                break
            time.sleep(0.1)
        seen["interview_after"] = drive.view()["interview"]
    return seen


def test_the_journey_reached_the_cards_it_scans(served):
    assert served["estimand"]["estimand"]["effects"] and served["adjustment"]["adjustment"]
    assert any(k.startswith("refused:") and served[k].get("error") for k in served)


def test_a_served_card_speaks_plainly(served):
    bad = {key: impolite(value) for key, value in served.items() if impolite(value)}
    assert not bad, bad


def test_the_plain_cards_stay_within_the_word_budgets_of_the_words_they_replace(served):
    """Plain phrases run longer than the terms ("what you study" for "exposure"); the questions and
    the options' consequences they now carry still fit the budgets the teaching holds them to."""
    B = teaching.BUDGETS
    estimand, adjustment = served["estimand"]["estimand"], served["adjustment"]["adjustment"]
    problems = []
    for option in (*estimand["effects"], *estimand["contrasts"]):
        if voice.words(option["consequence"]) > B["option_consequence"]:
            problems.append(("consequence", option["consequence"]))
    for key, question in adjustment["questions"].items():
        if voice.words(question) > B["question"]:
            problems.append((key, question))
    assert not problems, problems


def test_the_refusals_are_real_refusals_in_plain_words(served):
    for key in (k for k in served if k.startswith("refused:")):
        error = served[key].get("error")
        assert error and error.get("message"), (key, served[key])
        assert plain.is_plain(error["message"]), (key, error["message"])
        for exit_ in error.get("exits") or []:
            assert plain.is_plain(exit_.get("label", "")), (key, exit_)
