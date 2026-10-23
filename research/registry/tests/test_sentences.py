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
* A read takes the quest log, the methods lines and the manuscript after every answer, and each
  stage artifact once for each content it serves, so an artifact that changes under its key (the
  fit, once the held-out rows open) is read again, and one that does not is not.
* The teaching entries read are those of the questions answered, the roles card under the goal,
  and the repairs entry only where a noticing's card is drawn; the drawer is counted, not kept.
* Two sentences that state one rule get verdicts that agree: alike but for their values, or one
  held whole in the other, a claim's sentence is never held by plain wording, and a registered
  claim's rule in other words is named by the sentence that holds it.
* The committed table is what the committed capture, the registry and the rulings give now, and
  the capture's render log traced the journey's claims this package registered: what a swarm
  record's ``claims`` lists; the capture holds the held-out results, the methods lines and the
  teaching of each answered question, and the table's verdicts on the sentences that state one
  rule twice agree.

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


class _Served:
    def __init__(self, body: object, status: int = 200):
        self.status_code, self.body = status, body

    def json(self) -> object:
        return self.body


class _Server:
    """A server's reads, as a capture makes them: each URL serves its body (a callable is asked
    each time), every other URL is not found; ``asked`` lists every URL read."""

    def __init__(self, routes: dict):
        self.routes, self.asked = routes, []

    def get(self, url: str, params: dict | None = None) -> _Served:
        key = url + ("?" + "&".join(f"{k}={v}" for k, v in sorted(params.items())) if params else "")
        self.asked.append(key)
        body = self.routes.get(key)
        if body is None:
            return _Served(None, 404)
        return _Served(body() if callable(body) else body)


BASE = "/api/projects/p1"


def _view(*answered: str, fit: str = "fresh") -> dict:
    return {"stages": {"fit": {"status": fit, "key": "same-key"}},
            "state": {"fit_pressed": False},
            "interview": [{"key": k, "status": "answered"} for k in answered]
            + [{"key": "event", "status": "open"}]}


def test_a_stage_artifact_changed_under_its_key_is_read_again_and_one_unchanged_is_not():
    """The fit artifact keeps its key when the held-out rows open, and its held-out sentence is new
    (a key read once would miss it); read again unchanged, it adds no read."""
    served = [{"result": {"sentence": "The held-out rows are sealed for now."}}]
    server = _Server({BASE: _view(), f"{BASE}/stages/fit": lambda: {"artifact": served[0]}})
    shown = S.Shown()
    S.read_shown(server, "p1", shown)
    served[0] = {"result": {"sentence": "The held-out rows were opened once and scored."}}
    S.read_shown(server, "p1", shown)
    assert {"The held-out rows are sealed for now.",
            "The held-out rows were opened once and scored."} <= set(shown.strings)
    kept = len(shown.snapshots)
    S.read_shown(server, "p1", shown)
    assert len(shown.snapshots) == kept  # the same content again: nothing new is read


def test_every_read_takes_the_methods_lines_and_the_manuscript_not_only_the_last():
    lines = {"lines": [{"record_id": "r1", "sentence": "The outcome was read as a regression."}]}
    sections = {"sections": [{"title": "Outcome", "entries": [
        {"text": "Glucose was the outcome, read as a regression."}]}]}
    server = _Server({BASE: _view("target"), f"{BASE}/methods": lines,
                      f"{BASE}/methods/sections": sections})
    shown = S.Shown()
    S.read_shown(server, "p1", shown)  # after an answer, not the final read
    assert shown.strings["The outcome was read as a regression."] == {
        ("methods:lines", "lines[].sentence")}
    assert ("methods", "sections[].entries[].text") in shown.strings[
        "Glucose was the outcome, read as a regression."]


def _entry(key: str, why: str, drawer: str | None = None) -> dict:
    return {"key": key, "title": key, "question": f"What is the {key} here?", "one_liner": "",
            "why": why, "consumer": "", "options": [], "terms": [],
            "drawer": {"sections": [{"heading": "In depth", "body": drawer}]} if drawer else None,
            "evidence": None}


def _teaching_server(findings: bool) -> _Server:
    plain = [_entry("lens", "A lens runs that field's checks first.",
                    drawer="The drawer says more. It says it in two sentences."),
             _entry("roles", "The roles card under no goal names a study factor."),
             _entry("event", "The event fixes what a probability means."),
             _entry("repairs", "A repair reads only its own row.")]
    goal = [*plain[:1], _entry("roles", "The roles card under prediction names predictors."),
            *plain[2:]]
    view = _view("lens", "roles")
    view["stages"]["findings"] = {"status": "fresh", "key": "k"}
    return _Server({BASE: view, "/api/teaching": plain, "/api/teaching?purpose=prediction": goal,
                    f"{BASE}/stages/findings": {"artifact": {
                        "findings": [{"id": "f1"}] if findings else []}}})


def test_the_teaching_read_is_each_answered_questions_with_the_repairs_where_a_noticing_shows():
    server = _teaching_server(findings=True)
    shown = S.Shown()
    S.read_shown(server, "p1", shown)
    left_out = S.read_teaching(server, shown, "prediction")
    texts = set(shown.strings)
    assert {"A lens runs that field's checks first.",
            "The roles card under prediction names predictors.",
            "A repair reads only its own row."} <= texts
    assert "The event fixes what a probability means." not in texts  # open, never answered
    assert "The roles card under no goal names a study factor." not in texts  # the goal's words
    assert not any("drawer" in t for t in texts)
    assert left_out == {"lens": 2}  # the drawer's two sentences, counted and not kept


def test_the_repairs_entry_is_not_read_where_no_noticing_card_is_drawn():
    server = _teaching_server(findings=False)
    shown = S.Shown()
    S.read_shown(server, "p1", shown)
    S.read_teaching(server, shown, "prediction")
    assert "A repair reads only its own row." not in shown.strings
    assert "A lens runs that field's checks first." in shown.strings


def _row(sentence: str, verdict: str, claims_: tuple = ()) -> "S.Row":
    return S.Row(sentence, [("quest", "x")], verdict, list(claims_), "")


def test_two_sentences_alike_but_for_their_values_disagree_unless_their_verdicts_match():
    a = _row("Read as binary, `male` becomes 1 and `female` becomes 0.", "plain")
    b = _row("Read as binary, `True` becomes 1 and `False` becomes 0.", "register")
    assert len(S.disagreements([a, b])) == 1
    assert S.disagreements([a, _row(b.sentence, "plain")]) == []
    assert S.disagreements([a, _row("A different sentence of its own words here.", "register")]) == []


def test_plain_wording_that_holds_a_claims_sentence_whole_disagrees_and_a_claim_holding_it_does_not():
    held = _row("a measurement spread over so narrow a span would repeat about `2,567` times.",
                "register")
    whole = ("Read from the values: `SEQN` is the unit's identifier (a measurement spread over so "
             "narrow a span would repeat about `2,567` times); `age` is a characteristic.")
    assert len(S.disagreements([held, _row(whole, "plain")])) == 1
    assert S.disagreements([held, _row(whole, "register")]) == []
    assert S.disagreements([_row(held.sentence, "plain"), _row(whole, "plain")]) == []
    short = _row("about `2,567` times.", "register")  # under six words: compared with none
    assert S.disagreements([short, _row(whole, "plain")]) == []


def test_a_sentence_holding_a_registered_claims_rule_in_other_words_names_that_claim():
    held = _row("How many days each value spans is not settled: record it before screening by "
                "`kcal`.", "unrendered", ("exclusions.screen_waits_for_days",))
    whole = ("How many days each value spans is not settled: record it before screening by "
             "`kcal`; `bmi` is an amount.")
    assert len(S.disagreements([held, _row(whole, "register")])) == 1
    assert S.disagreements([held, _row(whole, "unrendered",
                                       ("exclusions.screen_waits_for_days",))]) == []
    assert S.disagreements([held, _row(whole, "claim", ("explore.collinear_pair",))]) == []


def test_the_table_refuses_verdicts_that_disagree(tmp_path):
    path = _capture(tmp_path, "Read as binary, `male` becomes 1 and `female` becomes 0.",
                    "Read as binary, `True` becomes 1 and `False` becomes 0.")
    split = [Ruling(r"`male`", "plain", "a reading"), Ruling(r"^Read as binary", "register", "BINARY")]
    assert S.table("planted", path, tmp_path / "planted.md", rulings=split) == 1
    alike = [Ruling(r"^Read as binary", "register", "BINARY")]
    assert S.table("planted", path, tmp_path / "planted.md", rulings=alike) == 0


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


def test_the_committed_capture_holds_the_held_out_results_the_methods_lines_and_the_teaching():
    """What the first capture left out (the repair round's P1): the fit's held-out detail after the
    opening, the methods line under each answer, and each answered question's teaching entry."""
    data = json.loads((S.HERE / "nhanes-glucose-predict.capture.json").read_text("utf-8"))
    places = {(s, p) for e in data["strings"] for s, p in e["shown"]}
    texts = {x for e in data["strings"] for x in S.sentences_of(e["text"])}
    assert any("holdout_detail" in p for _, p in places)
    assert any(t.startswith("The interval is the held-out R²'s (influence function") for t in texts)
    assert ("methods:lines", "lines[].sentence") in places
    answered = sorted(k for k, v in data["interview"].items() if v == "answered")
    assert data["teaching"]["read"] == sorted([*answered, "repairs"])
    assert {s for s, _ in places if s.startswith("teaching:")} == {
        f"teaching:{k}" for k in data["teaching"]["read"]}
    assert sum(data["teaching"]["left_out"]["sentences"].values()) > 0


def test_the_committed_tables_verdicts_on_one_rule_said_twice_agree():
    """The repair round's P2: each sentence that states a rule another sentence's verdict names
    takes that verdict, not plain wording."""
    data = json.loads((S.HERE / "nhanes-glucose-predict.capture.json").read_text("utf-8"))
    rows = S.rows_of(data, claims.registry(), RULINGS)
    assert S.disagreements(rows) == []

    def verdict(prefix: str) -> tuple[str, list[str]]:
        found = {(r.verdict, tuple(r.claims)) for r in rows if r.sentence.startswith(prefix)}
        assert len(found) == 1, (prefix, found)
        v, c = found.pop()
        return v, list(c)

    assert verdict("Read from the values, no question asked: `SEQN` is the unit's identifier")[0] \
        == "register"
    assert "exclusions.screen_waits_for_days" in verdict(
        "How many days each value spans is not settled: its median, 1,950, is an adult's day's "
        "intake (a child's total over two days sits there too); record it before screening by "
        "`kcal`); `bp_sys`")[1]
    assert verdict("Read as binary, male becomes 1 and female becomes 0.")[0] == "register"
    assert verdict("Sound: a near-collinear pair does not harm prediction") == (
        "unrendered", ["materiality.collinear_under_prediction"])
    assert verdict("At the opening (decision #")[0] == "register"
