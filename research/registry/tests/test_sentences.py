"""The journey's sentences (``research/registry/sentences.py``; R1_PLAN.md R1-B): what the capture
reads as a sentence, how a string splits, how a sentence is indexed to the claim that rendered it
or triaged, and the committed table held current.

* A sentence is a string of at least four words that is not a name; the capture skips the values
  of keys that name things, and keeps every place a string was served, its list indices dropped.
* A string splits at a sentence's end, never after "et al.", and a Markdown line's marks drop.
* A sentence that holds a claim's words, or one clause of them, is indexed to it; one that does
  not is triaged by the first ruling that matches it, else untriaged, and the table exits 1.
* A sentence served only on the card of a question the journey does not ask is not shown; one
  served anywhere else is.
* The committed table is what the committed capture, the registry and the rulings give now, and
  the capture's render log traced the journey's claims this package registered: what a swarm
  record's ``claims`` lists.

Every expectation is built here, or read from the committed files the test holds current.
"""
from __future__ import annotations

import json
from pathlib import Path

from research.registry import sentences as S
from research.registry.triage import RULINGS, Ruling
from turbotab.core import claims


def test_a_string_is_a_sentence_only_with_four_words_and_not_as_a_name():
    assert S.is_sentence("The fit waits for it.")
    assert not S.is_sentence("Keep every row")  # a label of three words
    assert not S.is_sentence("imputed_bp_sys")
    assert not S.is_sentence("2026-10-22T19:56:08+00:00 at the start of it")
    assert not S.is_sentence('{"kind": "set_split", "holdout": 0.2}')


def test_the_capture_keeps_words_under_word_keys_and_skips_names_with_every_place():
    node = {"id": "a long identifier with spaces in it", "stages": [
        {"lines": [{"reason": "This line says why it waits.", "column": "a column name here"}]},
        {"lines": [{"reason": "This line says why it waits."}]}]}
    shown = S.Shown()
    shown.add(node, "quest")
    shown.add({"why": "This line says why it waits."}, "artifact:fit")
    assert shown.strings == {"This line says why it waits.": {
        ("quest", "stages[].lines[].reason"), ("artifact:fit", "why")}}


def test_a_string_splits_at_each_sentence_and_not_after_an_abbreviation():
    text = ("## Missing data (TRIPOD+AI 11)\n\nA fill in each fold (Sisk et al. 2023) scores a "
            "new row. Complete cases cannot score one; e.g. a blank. A third one is here.")
    assert S.sentences_of(text) == [
        "Missing data (TRIPOD+AI 11)",
        "A fill in each fold (Sisk et al. 2023) scores a new row.",
        "Complete cases cannot score one; e.g. a blank.",
        "A third one is here."]


def test_a_sentence_holding_a_claims_words_or_one_clause_of_them_is_indexed():
    said = [S.Said("the fit waits for the answer for each.", "a.whole", "card"),
            S.Said("one clause of a claim's card", "b.split", "card"),
            S.Said("First sentence of a claim. Second sentence of a claim.", "c.two", "card")]
    assert S.index("`x` holds whole numbers, which may be codes; the fit waits for the answer "
                   "for each.", said) == ["a.whole"]
    assert S.index("Its option: one clause of a claim's card, and more.", said) == ["b.split"]
    assert S.index("Second sentence of a claim.", said) == ["c.two"]
    assert S.index("A sentence no claim says anywhere at all.", said) == []


def test_the_registrys_sentences_include_a_claims_clauses_placed_apart():
    """``explore.missing_by_group`` renders its card's two clauses apart, each indexed."""
    said = S.claim_sentences(claims.registry())
    mine = {s.text for s in said if s.claim == "explore.missing_by_group"}
    assert {"Multiple imputation fills blanks from the analysis's variables",
            "complete cases state their assumption"} <= mine


def _capture(tmp_path: Path, *texts: str, interview: dict | None = None) -> Path:
    data = {"journey": {"name": "planted", "words": "A planted journey."},
            "captured": {"date": "2026-10-22", "commit": "0", "seconds": 1, "export": 200},
            "interview": interview or {},
            "rendered": [{"id": "explore.collinear_pair", "register": "card", "renders": 1}],
            "strings": [{"text": t, "shown": [["quest", "stages[].lines[].reason"]]}
                        for t in texts]}
    path = tmp_path / "planted.capture.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_an_unindexed_sentence_is_triaged_by_the_first_ruling_else_the_table_refuses(tmp_path):
    words = "3 pairs of predictors move almost together (|r| ≥ 0.9)"
    path = _capture(tmp_path, words, "Your change to Models made 1 result out of date.",
                    "A sentence no ruling covers at all.")
    rulings = [Ruling(r"^Your change to", "plain", "says what an answer reopened"),
               Ruling(r"^Your change", "register", "never reached: the first ruling decides")]
    rows = {r.sentence: r for r in S.rows_of(json.loads(path.read_text()), claims.registry(),
                                               rulings)}
    assert (rows[words].verdict, rows[words].claims) == ("claim", ["explore.collinear_pair"])
    assert rows["Your change to Models made 1 result out of date."].verdict == "plain"
    assert rows["A sentence no ruling covers at all."].verdict == "untriaged"
    out = tmp_path / "planted.md"
    assert S.table("planted", path, out, rulings=rulings) == 1
    assert S.table("planted", path, out, rulings=[*rulings, Ruling(r"^A sentence", "plain",
                                                                   "copy")]) == 0
    assert "| `explore.collinear_pair` | 1 | yes |" in out.read_text(encoding="utf-8")


def test_a_ruling_that_names_an_unregistered_claim_refuses_the_table(tmp_path):
    path = _capture(tmp_path, "A sentence a ruling covers here.")
    rulings = [Ruling(r"^A sentence", "unrendered", "said elsewhere", ("no.such_claim",))]
    assert S.table("planted", path, tmp_path / "planted.md", rulings=rulings) == 1


def test_check_refuses_a_stale_table_and_passes_a_current_one(tmp_path):
    path = _capture(tmp_path, "A sentence a ruling covers here.")
    rulings = [Ruling(r"^A sentence", "plain", "copy")]
    out = tmp_path / "planted.md"
    assert S.table("planted", path, out, check=True, rulings=rulings) == 1  # none written yet
    assert S.table("planted", path, out, rulings=rulings) == 0
    assert S.table("planted", path, out, check=True, rulings=rulings) == 0
    out.write_text(out.read_text(encoding="utf-8") + "an edit by hand\n", encoding="utf-8")
    assert S.table("planted", path, out, check=True, rulings=rulings) == 1


def test_a_sentence_only_on_an_unasked_questions_card_is_not_shown_and_elsewhere_it_is():
    unasked = {"energy_adjustment": "not_applicable"}
    energy = [("artifact:proposals", "labels.energy_adjustment.options[]")]
    assert S.not_shown(energy, unasked) == "energy_adjustment"
    assert S.not_shown(energy, {"energy_adjustment": "answered"}) is None
    assert S.not_shown([*energy, ("quest", "stages[].lines[].reason")], unasked) is None
    assert S.not_shown([("artifact:proposals", "labels.missing.tension")], unasked) is None


def test_the_committed_table_is_current():
    assert S.table("nhanes-glucose-predict", check=True) == 0


def test_the_committed_capture_traces_the_claims_this_package_registered():
    """Each refusal and default the package registered renders in the journey where the claims
    render log reads it, so the capture's log, what a swarm record's ``claims`` lists, holds it."""
    data = json.loads((S.HERE / "nhanes-glucose-predict.capture.json").read_text("utf-8"))
    traced = {r["id"] for r in data["rendered"]}
    assert {"readings.whole_numbers_wait", "missing.blank_not_asked",
            "missing.fold_fill_first_under_prediction",
            "exclusions.every_row_first_under_prediction", "exclusions.screen_waits_for_days",
            "explore.collinear_pair"} <= traced
    assert data["captured"]["export"] == 200
    assert all(r.verdict in ("plain", "register", "unrendered") for r in RULINGS)
