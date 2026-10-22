"""What the driver offers for an open question (``research.swarm.options``), on artifacts built here
in the shapes the server serves (``turbotab/server/schemas.py``'s models): every option the
artifact offers is listed by the id the server serves, in the server's words and none of the
driver's own; an id or a field it does not offer is refused; ``--set`` changes only what it names.

The expected ids and decisions are read off the constructed artifacts by hand, and a way
forward's id is derived here from its decision with hashlib, independently of the module."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from research.swarm import options as opts


def served_from(artifacts: dict[str, Any]):
    return lambda stage: artifacts.get(stage)


def view(**state: Any) -> dict[str, Any]:
    return {"state": state, "interview": [], "stages": {}, "decisions": []}


def compose(key: str, artifacts: dict[str, Any], entry: dict[str, Any] | None = None,
            step: dict[str, Any] | None = None, **state: Any) -> dict[str, Any]:
    return opts.compose(key, step or {"key": key, "status": "open"}, view(**state),
                        served_from(artifacts), entry)


def strings(obj: Any, keys=("label", "consequence", "note", "reason")) -> list[str]:
    """Every sentence-like string the composed question shows under ``keys``."""
    out: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys and isinstance(v, str):
                out.append(v)
            out += strings(v, keys)
    elif isinstance(obj, list):
        for v in obj:
            out += strings(v, keys)
    return out


def served_strings(*sources: Any) -> set[str]:
    out: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, str):
            out.add(node)
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    for s in sources:
        walk(s)
    return out


# ── constructed artifacts, in the server's shapes ─────────────────────────────

LABELED = lambda question, keys: {  # noqa: E731 - a LabeledQuestion (custom_sound)
    "question": question, "purpose": "prediction", "customary_first": keys[0], "tension": None,
    "options": [{"key": k, "label": f"{k} as the card says it",
                 "customary": {"field": "f", "text": f"customary {k}", "source": "s"},
                 "sound": {"purpose": "prediction", "verdict": "sound", "reason": f"sound {k}"}}
                for k in keys]}

SEAL_PLAN = {
    "task": "binary", "n_measured": 480, "n_analyzed": 480, "basis": None, "chronology": None,
    "exploratory": False, "floor": {"text": "A held-out score needs 100 events."},
    "options": [{"holdout": 0.0, "label": "Cross-validation only", "n_holdout": 0,
                 "measures": "No held-out score.", "below_floor": False},
                {"holdout": 0.2, "label": "Hold out 20%", "n_holdout": 96,
                 "measures": "A score within about 0.1.", "below_floor": True}],
    "cv_first": True, "reason": "Cross-validation reuses every row.", "precision_note": "Widths.",
    "refusal": None,
    "validation": {"options": [
        {"validation": "bootstrap", "label": "Bootstrap optimism", "measures": "m", "cost": "c"},
        {"validation": "kfold", "label": "k-fold", "measures": "m2", "cost": "c2"},
        {"validation": "internal_external", "label": "By site", "measures": "m3", "cost": "c3",
         "needs_cluster": True, "cluster": "site"}],
        "resampling_first": True, "reason": "Below 500 rows resampling leads.", "holdout_note": None}}

ROLES = {"columns": [
    {"column": "encounter_id", "proposed": "identifier", "confidence": "medium", "reason": "Names each record."},
    {"column": "age", "proposed": "covariate", "confidence": "high", "reason": "A characteristic."},
    {"column": "los", "proposed": "exposure", "confidence": "low", "reason": "A measured column."}],
    "needs_confirmation": ["encounter_id", "los"]}
ROLES_ENTRY = {"key": "roles", "question": "What role does each column play?", "options": [
    {"value": v, "label": v.title(), "consequence": f"{v} enters as such"}
    for v in ("identifier", "exposure", "covariate", "excluded")]}

ADJUSTMENT = {"exposure": "sugar", "effect": "total",
              "questions": {"causes_exposure": "Does it cause sugar?", "causes_outcome": "Glucose?",
                            "after_exposure": "After?"},
              "groups": [{"key": "demographics", "label": "Age and sex", "columns": ["age", "sex"],
                          "guess": {"causes_exposure": "yes", "causes_outcome": "yes",
                                    "after_exposure": "no"},
                          "reason": "They come before the diet.", "derived": "confounder",
                          "derived_words": "confounders",
                          "decision": {"kind": "set_adjustment", "exposure": "sugar", "answers": {
                              c: {"causes_exposure": "yes", "causes_outcome": "yes",
                                  "after_exposure": "no"} for c in ("age", "sex")}}},
                         {"key": "unguessed", "label": "Not guessed", "columns": ["bmi"],
                          "guess": None, "reason": "Nothing to guess from.", "derived": None,
                          "derived_words": None, "decision": None}],
              "answered": {}, "adjusted": [], "left_out": [], "secondary": [], "source": "pack"}

FORMS = {"purpose": "inference", "ready": True, "exposure": "sugar", "task": "regression",
         "n_effective": 900.0, "rule_knots": 4, "rule": "Harrell: 4 knots",
         "needs": [{"column": "sugar", "role": "exposure", "receives": "sugar", "scale": "",
                    "unit": "g", "zero_share": 0.0, "mass_at_zero": False, "n_effective": 900.0,
                    "rule_knots": 4, "rule": "Harrell: 4 knots",
                    "options": [{"value": v, "label": v, "customary": "c", "sound": "s",
                                 "consequence": f"{v} form", "rung": "r"}
                                for v in ("spline", "linear", "quintiles")],
                    "proposal": {"form": "spline", "knots": 4, "knots_rule": "harrell"}}],
         "stated": [], "waiting": [],
         "answer": {"kind": "set_forms", "forms": {"sugar": {"form": "spline", "knots": 4,
                                                             "knots_rule": "harrell"}}},
         "beside": None}

SHELF = {"families": [{"key": "ridge", "label": "Ridge", "rank": 2, "fit": "ok", "concerns": [],
                       "inductive_bias": "Shrinks."},
                      {"key": "linear", "label": "Linear model", "rank": 1, "fit": "ok",
                       "concerns": [], "inductive_bias": "A line."}],
         "basis": "Ranked for 384 rows."}


# ── ids ──────────────────────────────────────────────────────────────────────


def test_a_way_forward_is_named_by_its_decisions_digest():
    """The id is ``x-`` and the first 8 hex digits of the SHA-256 of the decision's canonical JSON
    (sorted keys, no spaces), derived here with hashlib: the same decision in another key order has
    the same id, another decision another, and a way with no decision has none."""
    decision = {"kind": "confirm_reading", "reading": "code_or_count", "column": "age", "value": "amount"}
    text = '{"column":"age","kind":"confirm_reading","reading":"code_or_count","value":"amount"}'
    expected = "x-" + hashlib.sha256(text.encode()).hexdigest()[:8]
    assert opts.exit_id(decision) == expected
    assert opts.exit_id(dict(reversed(list(decision.items())))) == expected
    assert opts.exit_id({**decision, "value": "code"}) != expected
    listed = opts.exits([{"label": "Wait for the fit", "decision": None},
                         {"label": "`age` is an amount", "decision": decision}])
    assert listed == [{"id": None, "label": "Wait for the fit", "decision": None},
                      {"id": expected, "label": "`age` is an amount", "decision": decision}]


# ── every option, by the server's id, in the server's words ──────────────────


def test_the_split_offers_each_holdout_and_its_validation_as_a_field():
    q = compose("split", {"seal_plan": SEAL_PLAN})
    # each holdout the plan serves, then a cut-point of one's own (the next test)
    assert [o["id"] for o in q["options"]] == ["0", "0.2", "cut"]
    assert [o["label"] for o in q["options"]][:2] == ["Cross-validation only", "Hold out 20%"]
    assert q["options"][1]["decision"] == {"kind": "set_split", "holdout": 0.2,
                                           "validation": "bootstrap"}  # the card's first scheme
    field = q["fields"][0]
    assert [f.get("applies_to") for f in q["fields"]] == [None, ["cut"], ["cut"], ["cut"]]
    assert field["name"] == "validation" and field["initial"] == "bootstrap"
    assert [c["validation"] for c in field["choices"]] == ["bootstrap", "kfold", "internal_external"]
    assert opts.resolve(q, ["0.2"], ["validation=kfold"]) == {
        "kind": "set_split", "holdout": 0.2, "validation": "kfold"}
    # a scheme that folds by a named grouping carries it
    assert opts.resolve(q, ["0.2"], ["validation=internal_external"]) == {
        "kind": "set_split", "holdout": 0.2, "validation": "internal_external", "cluster": "site"}
    with pytest.raises(opts.Unanswerable):
        opts.resolve(q, ["0.2"], ["validation=leave_one_out"])


SPLIT_ENTRY = {"key": "split", "options": [
    {"value": "0", "label": "Cross-validation only", "consequence": "No rows are sealed."},
    {"value": "cut", "label": "Hold out by a cut-point",
     "consequence": "Every row at or after a value you name is sealed."}]}
TIME_COLUMNS = {"columns": [{"name": "__row_id", "dtype": "integer"},
                            {"name": "cycle", "dtype": "integer"},
                            {"name": "visit", "dtype": "datetime"},
                            {"name": "sex", "dtype": "categorical"},
                            {"name": "LBXGLU", "dtype": "numeric"}]}


def test_a_cut_point_of_ones_own_sets_its_column_its_value_and_its_folds_order():
    """SWARM_PLAN §8: the glucose project holds out its later cycle by its cycle column. The
    option records nothing until its fields are set; the column is one that can read as time
    (never the outcome, never the row id, never a column of levels); the value is passed as typed,
    for the engine to read as the column reads; the folds' order is random or time, unchosen."""
    q = compose("split", {"seal_plan": SEAL_PLAN, "ingest": TIME_COLUMNS}, SPLIT_ENTRY,
                target="LBXGLU")
    cut = next(o for o in q["options"] if o["id"] == "cut")
    assert (cut["label"], cut["consequence"]) == (
        "Hold out by a cut-point", "Every row at or after a value you name is sealed.")
    assert cut["decision"] is None
    fields = {f["name"]: f for f in q["fields"]}
    assert [c["value"] for c in fields["cut_column"]["choices"]] == ["cycle", "visit"]
    assert [c["value"] for c in fields["fold_order"]["choices"]] == ["random", "time"]
    assert all(fields[n]["initial"] is None for n in ("cut_column", "cut_at", "fold_order"))
    assert opts.resolve(q, ["cut"], ["cut_column=cycle", "cut_at=2017", "fold_order=random",
                                     "validation=kfold"]) == {
        "kind": "set_split", "holdout": 0.0, "validation": "kfold",
        "cut_point": {"column": "cycle", "at": "2017"}, "fold_order": "random"}
    with pytest.raises(opts.Unanswerable):
        opts.resolve(q, ["cut"], [])  # nothing set: it records nothing
    with pytest.raises(opts.Unanswerable):
        opts.resolve(q, ["cut"], ["cut_column=sex", "cut_at=2017", "fold_order=random"])
    with pytest.raises(opts.Unanswerable):
        opts.resolve(q, ["cut"], ["cut_column=cycle", "cut_at=2017", "fold_order=shuffled"])
    with pytest.raises(opts.Unanswerable):
        opts.resolve(q, ["0.2"], ["cut_at=2017"])  # a cut-point's field on a share


def test_what_the_cut_point_option_records_the_engine_accepts():
    """Silent: the decision ``resolve`` builds parses as the engine's split, its value read as the
    column reads it."""
    from turbotab.core.decisions import parse_decision, cut_shown

    q = compose("split", {"seal_plan": SEAL_PLAN, "ingest": TIME_COLUMNS}, SPLIT_ENTRY,
                target="LBXGLU")
    decision = parse_decision(opts.resolve(q, ["cut"], ["cut_column=cycle", "cut_at=2017",
                                                        "fold_order=time"]))
    assert decision.cut_point.column == "cycle" and cut_shown(decision.cut_point.at) == "2017"
    assert decision.holdout == 0.0 and decision.fold_order == "time"


def test_the_roles_are_the_proposal_with_one_field_per_column():
    q = compose("roles", {"roles": ROLES}, ROLES_ENTRY, purpose="prediction")
    assert [o["id"] for o in q["options"]] == ["proposed"]
    proposal = {"encounter_id": "identifier", "age": "covariate", "los": "exposure"}
    assert q["options"][0]["decision"] == {"kind": "set_roles", "roles": proposal}
    assert {f["name"]: f["initial"] for f in q["fields"]} == proposal
    assert opts.resolve(q, ["proposed"], ["los=excluded"]) == {
        "kind": "set_roles", "roles": {**proposal, "los": "excluded"}}
    assert q["options"][0]["decision"]["roles"]["los"] == "exposure"  # the offer itself unchanged
    for bad in (["los=predictor"], ["weight=covariate"], ["los"]):
        with pytest.raises(opts.Unanswerable):
            opts.resolve(q, ["proposed"], bad)


def test_the_adjustment_offers_each_groups_one_tap_and_answers_per_column():
    q = compose("adjustment", {"proposals": {"adjustment": ADJUSTMENT}}, purpose="inference")
    assert [o["id"] for o in q["options"]] == ["demographics", "answers"]
    assert q["options"][0]["decision"] == ADJUSTMENT["groups"][0]["decision"]
    assert q["data"]["questions"] == ADJUSTMENT["questions"]
    assert opts.resolve(q, ["answers"], ["bmi=unknown,yes,unknown", "age=yes,yes,no"]) == {
        "kind": "set_adjustment", "exposure": "sugar",
        "answers": {"bmi": {"causes_exposure": "unknown", "causes_outcome": "yes",
                            "after_exposure": "unknown"},
                    "age": {"causes_exposure": "yes", "causes_outcome": "yes", "after_exposure": "no"}}}
    with pytest.raises(opts.Unanswerable):
        opts.resolve(q, ["answers"], [])  # records nothing until a column is answered
    for bad in ("bmi=yes,no", "bmi=maybe,yes,no", "bmi=yes,yes,no,no"):
        with pytest.raises(opts.Unanswerable):
            opts.resolve(q, ["answers"], [bad])
    with pytest.raises(opts.Unanswerable):  # the per-column answers belong to their option
        opts.resolve(q, ["demographics"], ["bmi=yes,yes,no"])


def test_the_forms_are_the_cards_one_tap_with_a_field_per_column():
    q = compose("form", {"forms": FORMS}, purpose="inference")
    assert q["options"] == [{"id": "proposed", "label": None, "decision": FORMS["answer"]}]
    assert opts.resolve(q, ["proposed"], ["sugar=linear"]) == {
        "kind": "set_forms", "forms": {"sugar": {"form": "linear"}}}
    # back to the proposal's form: the proposal itself, its k by the card's rule
    assert opts.resolve(q, ["proposed"], ["sugar=spline"]) == FORMS["answer"]
    with pytest.raises(opts.Unanswerable):
        opts.resolve(q, ["proposed"], ["sugar=cubic"])


def test_the_lens_and_the_models_take_several_options_together():
    shelf = compose("models", {"shelf": SHELF})
    assert [o["id"] for o in shelf["options"]] == ["linear", "ridge"]  # the shelf's rank order
    assert opts.resolve(shelf, ["linear", "ridge"], []) == {"kind": "select_models",
                                                             "models": ["linear", "ridge"]}
    entry = {"key": "lens", "options": [{"value": "dietary", "label": "Dietary intake", "consequence": "c1"},
                                        {"value": "clinical", "label": "Clinical", "consequence": "c2"}]}
    lens = compose("lens", {}, entry)
    assert opts.resolve(lens, ["clinical", "dietary"], []) == {"kind": "set_lens",
                                                               "lenses": ["clinical", "dietary"]}
    missing = compose("missing", {"proposals": {"labels": {"missing": LABELED(
        "missing", ["impute", "complete_case"])}}})
    with pytest.raises(opts.Unanswerable):  # one option at a time elsewhere
        opts.resolve(missing, ["impute", "complete_case"], [])


def test_an_id_not_offered_is_refused_and_a_way_forward_is_taken_by_its_id():
    q = compose("missing", {"proposals": {"labels": {"missing": LABELED(
        "missing", ["impute", "complete_case"])}}})
    assert [o["id"] for o in q["options"]] == ["impute", "complete_case"]
    with pytest.raises(opts.Unanswerable):
        opts.resolve(q, ["multiple_imputation"], [])
    with pytest.raises(opts.Unanswerable):
        opts.resolve(q, [], [])
    way = {"kind": "confirm_readings", "items": [{"reading": "code_or_count", "column": "age",
                                                  "value": "amount"}]}
    ways = opts.exits([{"label": "Confirm each as shown", "decision": way},
                       {"label": "Wait", "decision": None}])
    assert opts.resolve(q, [ways[0]["id"]], [], ways) == way
    assert opts.resolve(q, ["impute"], [], ways)["kind"] == "set_missing"


# The exclusions card as the app shows it (ExclusionsAsk): keeping every row, each screen offered
# on this table, and a range of one's own on a numeric column, in the teaching entry's words.
EXCLUSIONS_ENTRY = {"key": "exclusions", "question": "Is your study restricted to part of this data?",
                    "options": [{"value": "none", "label": "Keep every row",
                                 "consequence": "No row is excluded; the record says so."},
                                {"value": "sex_neutral_500_3500", "label": "500–3,500 kcal a day",
                                 "consequence": "Anyone outside 500–3,500 kcal a day is excluded."},
                                {"value": "custom", "label": "Your own range",
                                 "consequence": "Rows outside a range you set are excluded."}]}
SCREEN = {"kind": "range", "column": "kcal", "low": 500.0, "high": 3500.0, "by": None,
          "reason": "implausible intake", "missing": "exclude"}
COLUMNS = {"columns": [{"name": "__row_id", "dtype": "integer"}, {"name": "kcal", "dtype": "numeric"},
                       {"name": "age", "dtype": "integer"}, {"name": "sex", "dtype": "categorical"},
                       {"name": "glucose", "dtype": "numeric"}, {"name": "smoker", "dtype": "boolean"}]}


def exclusions(labeled: bool = True) -> dict[str, Any]:
    proposals: dict[str, Any] = {
        "exclusions": [{"key": "sex_neutral_500_3500", "label": "500 to 3,500 kcal", "rule": SCREEN,
                        "refused": "Waits for the energy unit.", "affected": 12}],
        "n_base": 900}
    if labeled:  # custom_sound.exclusions: keeping every row, then the screens offered
        proposals["labels"] = {"exclusions": LABELED("exclusions",
                                                     ["keep_every_row", "sex_neutral_500_3500"])}
    return compose("exclusions", {"proposals": proposals, "oriented": COLUMNS}, EXCLUSIONS_ENTRY,
                   target="glucose")


@pytest.mark.parametrize("labeled", [True, False])
def test_the_exclusions_card_offers_every_row_each_screen_and_a_range_of_ones_own(labeled):
    """As the card offers them: keeping every row, in the teaching entry's words, under its id;
    each screen offered with the rule it records; a range of one's own, which records nothing
    until it is set."""
    q = exclusions(labeled)
    assert [o["id"] for o in q["options"]] == ["none", "sex_neutral_500_3500", "custom"]
    none, screen, custom = q["options"]
    assert (none["label"], none["consequence"]) == ("Keep every row",
                                                     "No row is excluded; the record says so.")
    assert none["decision"] == {"kind": "set_exclusions", "rules": []}
    assert screen["decision"] == {"kind": "set_exclusions", "rules": [SCREEN]}
    assert screen["label"] == "500–3,500 kcal a day" and screen["note"] == "Waits for the energy unit."
    assert (custom["label"], custom["decision"]) == ("Your own range", None)
    if labeled:  # the two labels the server gives keeping every row ride on its option
        assert none["labels"]["customary"]["text"] == "customary keep_every_row"
    else:
        assert "labels" not in none
    fields = {f["name"]: f for f in opts.public(q)["fields"]}
    assert set(fields) == {"column", "low", "high"}
    # the numeric columns, never the outcome or the row id (AskFor.tsx's numericColumns)
    assert [c["value"] for c in fields["column"]["choices"]] == ["kcal", "age"]
    assert all(f["applies_to"] == ["custom"] for f in fields.values())


def test_a_range_of_ones_own_records_the_rule_the_card_records():
    """Set, it records the app's own range rule (ChoiceQuestions.tsx's customRule): the column,
    the bounds given (one may be open), by none, the card's reason, and a row with no value left
    out as not confirmed eligible."""
    q = exclusions()
    rule = {"kind": "range", "column": "age", "low": 18.0, "high": 95.5, "by": None,
            "reason": "implausible values (your own range)", "missing": "exclude"}
    assert opts.resolve(q, ["custom"], ["column=age", "low=18", "high=95.5"]) == {
        "kind": "set_exclusions", "rules": [rule]}
    assert opts.resolve(q, ["custom"], ["column=kcal", "high=4000"])["rules"] == [
        {**rule, "column": "kcal", "low": None, "high": 4000.0}]
    assert opts.resolve(q, ["custom"], ["column=age", "low=-1e3"])["rules"][0]["low"] == -1000.0


@pytest.mark.parametrize("sets,word", [
    ([], "--set"),  # nothing set: it records nothing by itself
    (["column=age"], "low"),  # neither bound
    (["low=18", "high=60"], "column"),  # no column
    (["column=age", "low=eighteen"], "low"),
    (["column=age", "high="], "high"),
    (["column=age", "low=nan"], "low"),
    (["column=age", "high=inf"], "high"),
    (["column=sex", "low=1"], "column"),  # not numeric
    (["column=glucose", "low=1"], "column"),  # the outcome
    (["column=__row_id", "low=1"], "column"),
    (["column=height", "low=1"], "column"),  # not a column
])
def test_a_range_without_a_bound_or_with_one_that_is_not_a_number_is_refused(sets, word):
    q = exclusions()
    with pytest.raises(opts.Unanswerable) as caught:
        opts.resolve(q, ["custom"], sets)
    assert word in str(caught.value), (sets, str(caught.value))


def test_a_range_is_set_only_on_its_own_option():
    with pytest.raises(opts.Unanswerable):
        opts.resolve(exclusions(), ["none"], ["column=age", "low=18"])


def test_the_estimand_offers_each_exposure_with_the_cards_measures_as_a_field():
    card = {"exposures": [{"column": "sugar", "energy_contrast": True},
                          {"column": "fiber", "energy_contrast": False}],
            "effects": [{"effect": "total", "contrast": None, "label": "Total", "consequence": "All paths."}],
            "contrasts": [{"effect": None, "contrast": "substitution", "label": "Substitution",
                           "consequence": "In place of the rest."},
                          {"effect": None, "contrast": "addition", "label": "Addition",
                           "consequence": "On top."}],
            "measures": [{"measure": "risk_ratio", "label": "Risk ratio", "fitted": True, "reason": "r",
                          "rank": 2},
                         {"measure": "odds_ratio", "label": "Odds ratio", "fitted": True, "reason": "o",
                          "rank": 1},
                         {"measure": "hazard_ratio", "label": "Hazard ratio", "fitted": False,
                          "reason": "No follow-up.", "rank": None}],
            "family": None}
    q = compose("estimand", {"proposals": {"estimand": card}}, purpose="inference")
    assert [o["id"] for o in q["options"]] == ["sugar", "fiber"]
    assert q["options"][1]["decision"] == {"kind": "set_estimand", "exposure": "fiber", "family": False,
                                           "effect": "total", "contrast": None,
                                           "measure": "odds_ratio"}  # the card's first-ranked
    assert opts.resolve(q, ["sugar"], ["contrast=substitution", "measure=risk_ratio"])["contrast"] \
        == "substitution"
    with pytest.raises(opts.Unanswerable):  # the energy contrast is not fiber's
        opts.resolve(q, ["fiber"], ["contrast=addition"])
    with pytest.raises(opts.Unanswerable):  # a measure named but not fitted is not offered
        opts.resolve(q, ["fiber"], ["measure=hazard_ratio"])
    assert q["data"]["named_not_fitted"] == [{"measure": "hazard_ratio", "label": "Hazard ratio",
                                              "reason": "No follow-up."}]


# ── no words of the driver's own ─────────────────────────────────────────────

CASES = [
    ("split", {"seal_plan": SEAL_PLAN}, None, {}),
    ("roles", {"roles": ROLES}, ROLES_ENTRY, {"purpose": "prediction"}),
    ("adjustment", {"proposals": {"adjustment": ADJUSTMENT}}, None, {"purpose": "inference"}),
    ("form", {"forms": FORMS}, None, {"purpose": "inference"}),
    ("models", {"shelf": SHELF}, None, {}),
    ("missing", {"proposals": {"labels": {"missing": LABELED("missing", ["impute", "complete_case"])}}},
     None, {}),
    ("follow_up", {"target_info": {"column": "y", "task": "binary", "follow_up": []}},
     {"key": "follow_up", "options": [{"value": "same", "label": "Same for everyone", "consequence": "c"},
                                      {"value": "varies", "label": "Follow-up varies", "consequence": "d"}]},
     {"target": "y"}),
    ("purpose", {}, {"key": "purpose", "options": [{"value": "prediction", "label": "Prediction",
                                                    "consequence": "Judged on new rows."}]}, {}),
    ("exclusions", {"proposals": {"exclusions": [{"key": "sex_neutral_500_3500", "label": "a screen",
                                                  "rule": SCREEN, "refused": "Waits."}],
                                  "labels": {"exclusions": LABELED("exclusions", ["keep_every_row"])}},
                    "oriented": COLUMNS}, EXCLUSIONS_ENTRY, {"target": "glucose"}),
]


@pytest.mark.parametrize("key,artifacts,entry,state", CASES, ids=[c[0] for c in CASES])
def test_every_word_shown_is_the_servers(key, artifacts, entry, state):
    """Each label, consequence, note and reason the composed question shows is found verbatim in
    what the server served (the artifacts and the teaching entry)."""
    q = compose(key, artifacts, entry, **state)
    shown = strings(opts.public(q))
    assert shown, key
    served = served_strings(artifacts, entry or {})
    assert [s for s in shown if s not in served] == []


def test_the_words_check_fires_on_a_composer_that_adds_words(monkeypatch):
    """The check above, the other way: a composer that writes its own label is caught."""
    def chatty(ctx):
        return {"options": [opts._option("x", {"kind": "set_purpose", "purpose": "prediction"},
                                         label="Pick this one, it is best")]}

    monkeypatch.setitem(opts.COMPOSERS, "purpose", chatty)
    q = compose("purpose", {}, {"key": "purpose", "options": []})
    assert [s for s in strings(q) if s not in served_strings({})] == ["Pick this one, it is best"]


def test_the_private_parts_are_never_printed():
    q = compose("split", {"seal_plan": SEAL_PLAN})
    assert any("_apply" in f for f in q["fields"])
    assert "_apply" not in json.dumps(opts.public(q))
