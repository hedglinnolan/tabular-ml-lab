"""The NHANES glucose journey's claims reach the project's claims render log (SWARM_PLAN §10
package 6a; R1_PLAN.md R1-B): each refusal and default the journey shows renders through its claim
where the person reads it, inside the stage or the answer that shows it, so a swarm record's
``claims`` (the render log, as ``GET …/analyst-log`` serves it) traces the sentence; and where the
sentence is not shown, nothing is logged for it.

Each case renders inside ``render_log`` and reads the lines it wrote: the claim asked for, its
register, its frozen status, no fallback and no refusal. The silent case of each renders the same
code where the sentence does not appear and finds no line for the claim. This file names the claims
it traces, so no claim names it as its test (meta-test 2 reads only a claim's own test files).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbotab.core import custom_sound, readings
from turbotab.core.claims._registry import render_log
from turbotab.core.stages import explore, proposals
from turbotab.core.tests.test_journey_claims import SCREENS, _blank_reading, _state


def _logged(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]


def _ids(path: Path) -> list[str]:
    return [line["id"] for line in _logged(path)]


def _rendered_once(path: Path, claim: str) -> None:
    lines = [x for x in _logged(path) if x["id"] == claim]
    assert len(lines) == 1, _logged(path)
    line = lines[0]
    assert (line["register"], line["status"], line["fallback"], line["refused"]) == (
        "card", "built", None, None), line


def test_the_whole_number_refusal_is_logged_where_the_fit_asks_and_not_where_it_does_not(tmp_path):
    log = tmp_path / "claims_rendered.jsonl"
    with render_log(log):
        with pytest.raises(readings.Unsettled):
            readings.predictors_or_ask(_state({"grade": "covariate"}),
                                       {"grade": {"dtype": "integer", "n_unique": 5}})
    _rendered_once(log, "readings.whole_numbers_wait")
    quiet = tmp_path / "quiet.jsonl"
    with render_log(quiet):
        readings.predictors_or_ask(_state({"flag": "covariate"}),
                                   {"flag": {"dtype": "integer", "n_unique": 2}})
    assert "readings.whole_numbers_wait" not in _ids(quiet)


def test_the_not_asked_reading_is_logged_for_a_yes_no_answer_and_not_for_a_measurement(tmp_path):
    log = tmp_path / "claims_rendered.jsonl"
    with render_log(log):
        _blank_reading("answered", [True, False, True, False, True] + [None] * 5, "boolean")
    _rendered_once(log, "missing.blank_not_asked")
    quiet = tmp_path / "quiet.jsonl"
    with render_log(quiet):
        _blank_reading("waist", [81.5, 97.25, 88.0] + [None] * 7, "float")
    assert "missing.blank_not_asked" not in _ids(quiet)


@pytest.mark.parametrize("question, claim, kwargs", [
    ("missing", "missing.fold_fill_first_under_prediction", {}),
    ("exclusions", "exclusions.every_row_first_under_prediction", {"offered": SCREENS}),
])
def test_a_rankings_tension_is_logged_only_under_the_purpose_that_shows_it(tmp_path, question,
                                                                         claim, kwargs):
    log = tmp_path / "claims_rendered.jsonl"
    with render_log(log):
        custom_sound.labels_for(question, "prediction", **kwargs)
    _rendered_once(log, claim)
    quiet = tmp_path / "quiet.jsonl"
    with render_log(quiet):
        custom_sound.labels_for(question, "inference", **kwargs)
    assert claim not in _ids(quiet)


def test_the_screens_wait_for_the_days_is_logged_while_they_are_unsettled(tmp_path):
    log = tmp_path / "claims_rendered.jsonl"
    reading = {"unit": "kcal", "basis": "decision", "days": 1, "confirmed": False,
               "unit_settled": True, "days_unsettled": True,
               "days_reading": {"evidence": "its median, 1,950, is an adult's day's intake"}}
    with render_log(log):
        assert proposals.unit_refusal("kcal", reading)
    _rendered_once(log, "exclusions.screen_waits_for_days")
    quiet = tmp_path / "quiet.jsonl"
    frame = pd.DataFrame({"kcal": [1950.0, 2400.0]})
    with render_log(quiet):
        settled = proposals.energy_unit_reading(frame, "kcal", {"unit": "kcal", "days": 1})
        assert proposals.unit_refusal("kcal", settled) is None
    assert "exclusions.screen_waits_for_days" not in _ids(quiet)


def test_the_collinear_summary_is_logged_where_a_pair_is_named(tmp_path):
    log = tmp_path / "claims_rendered.jsonl"
    with render_log(log):
        text = explore.collinear_summary(3)
    assert text == "3 pairs of predictors move almost together (|r| ≥ 0.9)"
    _rendered_once(log, "explore.collinear_pair")
    quiet = tmp_path / "quiet.jsonl"
    x = np.arange(10.0)
    with render_log(quiet):
        assert explore._collinear(pd.DataFrame({"a": x, "b": np.sin(x)}), ["a", "b"]) == []
    assert _ids(quiet) == []
