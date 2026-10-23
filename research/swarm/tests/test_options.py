"""What the driver offers for an open question (``research.swarm.options``), on artifacts built here
in the shapes the server serves (``turbotab/server/schemas.py``'s models): every option the
artifact offers is listed by the id the server serves, in the server's words and none of the
driver's own; an id or a field it does not offer is refused; ``--set`` changes only what it names.

The expected ids and decisions are read off the constructed artifacts by hand, and a way
forward's id is derived here from its decision with hashlib, independently of the module. The last
test answers a level rule through the driver's own ``answer`` path on an in-process server and
checks the participant flow's counts against pandas."""
from __future__ import annotations

import hashlib
import json
import re
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
    # the plan's scheme is the shares'; the cut-point's fields, its own scheme among them (V5)
    assert [f.get("applies_to") for f in q["fields"]] == [["0", "0.2"], ["cut"], ["cut"], ["cut"],
                                                          ["cut"]]
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
                                     "cut_validation=kfold"]) == {
        "kind": "set_split", "holdout": 0.0, "validation": "kfold",
        "cut_point": {"column": "cycle", "at": "2017"}, "fold_order": "random"}
    with pytest.raises(opts.Unanswerable):  # the shares' scheme, worded for the shares' folds
        opts.resolve(q, ["cut"], ["cut_column=cycle", "cut_at=2017", "fold_order=random",
                                  "validation=kfold"])
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


def test_the_cut_points_scheme_carries_the_plans_words_only_for_the_folds_they_were_made_for():
    """V5: the seal plan words its validation choices for the folds the held-out draw orders
    (``time_ordered_folds``), and a cut-point's folds' order is its own answer, so its scheme's
    choices carry those words only under the order they were made for: random, where the plan's
    folds are not time-ordered; none where they are. The shares' field keeps the plan's words."""
    q = compose("split", {"seal_plan": SEAL_PLAN, "ingest": TIME_COLUMNS}, SPLIT_ENTRY,
                target="LBXGLU")
    fields = {f["name"]: f for f in q["fields"]}
    cut = fields["cut_validation"]
    assert cut["applies_to"] == ["cut"] and cut["initial"] == "bootstrap"
    assert [c["validation"] for c in cut["choices"]] == ["bootstrap", "kfold", "internal_external"]
    plan = SEAL_PLAN["validation"]["options"]
    assert [c["with_fold_order"] for c in cut["choices"]] == [
        {"random": {k: o[k] for k in ("label", "measures", "cost")}} for o in plan]
    assert all("label" not in c for c in cut["choices"])  # never at the top, read for any order
    timed = compose("split", {"seal_plan": {**SEAL_PLAN, "time_ordered_folds": True},
                              "ingest": TIME_COLUMNS}, SPLIT_ENTRY, target="LBXGLU")
    by_name = {f["name"]: f for f in timed["fields"]}
    assert [set(c) for c in by_name["cut_validation"]["choices"]] == [
        {"validation"}, {"validation"}, {"validation", "cluster"}]
    assert [c["label"] for c in by_name["validation"]["choices"]] == [o["label"] for o in plan]
    # the scheme set on the cut-point records as the shares' does, a grouping carried
    assert opts.resolve(timed, ["cut"], ["cut_column=cycle", "cut_at=2017", "fold_order=random",
                                         "cut_validation=internal_external"]) == {
        "kind": "set_split", "holdout": 0.0, "validation": "internal_external", "cluster": "site",
        "cut_point": {"column": "cycle", "at": "2017"}, "fold_order": "random"}


def basis(state: str, column: str | None) -> dict[str, Any]:
    """A seal plan's basis as ``seal.SealBasis`` serves it."""
    return {"state": state, "column": column, "label": state, "sentence": "",
            "exploratory": state in ("abandoned", "undetermined"), "source": "grain",
            "n_units": None}


@pytest.mark.parametrize("state, column", [("undetermined", None), ("abandoned", "pid"),
                                           ("abandoned", None)])
def test_the_cut_points_served_line_promises_no_whole_unit_where_the_seal_draws_by_row(state,
                                                                                    column):
    """The package 4 wording verifier's W1: the cut-point option's consequence is the served
    teaching entry's (``teaching.entries``, as ``GET /teaching`` serves it, never a stand-in), and
    under a basis that draws by row (undetermined; abandoned, by a column or by none) the seal
    holds the rows out one by one, a unit on both sides of the cut-point
    (``turbotab/core/tests/test_cut_point_seal.py`` draws it), so neither the served line, under
    any purpose, nor the driver's option promises a unit kept whole."""
    from turbotab.core import teaching

    served = {}
    for purpose in (None, "prediction", "inference"):
        entry = next(e for e in teaching.entries(purpose)
                     if e.key == "split").model_dump(mode="json")
        served[purpose] = next(o for o in entry["options"] if o["value"] == "cut")["consequence"]
        assert not re.search(r"whole|unit", served[purpose]), (purpose, served[purpose])
        q = compose("split", {"seal_plan": {**SEAL_PLAN, "basis": basis(state, column)},
                              "ingest": TIME_COLUMNS}, entry, target="LBXGLU")
        cut = next(o for o in q["options"] if o["id"] == "cut")
        assert cut["consequence"] == served[purpose]
    assert set(served.values()) == {"Rows at or after a time value you name are sealed."}


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
# on this table, a range of one's own on a numeric column, of plausible values or of who the study
# is about, and levels of one's own, in the teaching entry's words.
EXCLUSIONS_ENTRY = {"key": "exclusions", "question": "Is your study restricted to part of this data?",
                    "options": [{"value": "none", "label": "Keep every row",
                                 "consequence": "No row is excluded; the record says so."},
                                {"value": "sex_neutral_500_3500", "label": "500–3,500 kcal a day",
                                 "consequence": "Anyone outside 500–3,500 kcal a day is excluded."},
                                {"value": "custom", "label": "Your own range",
                                 "consequence": "Rows outside a range you set are excluded."},
                                {"value": "eligibility", "label": "Who it is about",
                                 "consequence": "Rows outside the study population leave."},
                                {"value": "levels", "label": "By level",
                                 "consequence": "Rows at levels you name are excluded."}]}
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
    each screen offered with the rule it records; a range of one's own, of plausible values or of
    who the study is about (the final fix round's V1), and levels of one's own (SWARM_PLAN §5, the
    package 4 verifier's P11), each recording nothing until it is set."""
    q = exclusions(labeled)
    assert [o["id"] for o in q["options"]] == ["none", "sex_neutral_500_3500", "custom",
                                               "eligibility", "levels"]
    none, screen, custom, eligibility, levels = q["options"]
    assert (none["label"], none["consequence"]) == ("Keep every row",
                                                     "No row is excluded; the record says so.")
    assert none["decision"] == {"kind": "set_exclusions", "rules": []}
    assert screen["decision"] == {"kind": "set_exclusions", "rules": [SCREEN]}
    assert screen["label"] == "500–3,500 kcal a day" and screen["note"] == "Waits for the energy unit."
    assert (custom["label"], custom["decision"]) == ("Your own range", None)
    assert (eligibility["label"], eligibility["consequence"], eligibility["decision"]) == (
        "Who it is about", "Rows outside the study population leave.", None)
    assert (levels["label"], levels["consequence"], levels["decision"]) == (
        "By level", "Rows at levels you name are excluded.", None)
    if labeled:  # the two labels the server gives keeping every row ride on its option
        assert none["labels"]["customary"]["text"] == "customary keep_every_row"
    else:
        assert "labels" not in none
    fields = {f["name"]: f for f in opts.public(q)["fields"]}
    assert set(fields) == {"column", "low", "high", "level_column", *LEVEL_LISTS, "missing"}
    # the numeric columns, never the outcome or the row id (AskFor.tsx's numericColumns)
    assert [c["value"] for c in fields["column"]["choices"]] == ["kcal", "age"]
    # a level rule reads any column but the outcome and the row id (AskFor.tsx's levelColumns)
    assert [c["value"] for c in fields["level_column"]["choices"]] == [
        "kcal", "age", "sex", "smoker"]
    assert {n: f["applies_to"] for n, f in fields.items()} == {
        **{n: ["custom", "eligibility"] for n in ("column", "low", "high")},
        **{n: ["levels"] for n in ("level_column", *LEVEL_LISTS)},
        "missing": ["custom", "eligibility", "levels"]}
    assert all(fields[n]["initial"] is None for n in ("level_column", *LEVEL_LISTS))


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
    for other in (["exclude=1"], ["level_column=sex"]):  # a level rule's field on the range
        with pytest.raises(opts.Unanswerable) as caught:
            opts.resolve(exclusions(), ["custom"], ["column=age", "low=18", *other])
        assert "does not apply to option custom" in str(caught.value)


# ── a range of who the study is about (the final fix round's V1) ─────────────
# A range of one's own records why its rows leave, and the participant flow and the figure the
# export draws from it print that reason beside the count ("Excluded: <reason>",
# turbotab/core/export/figures.py). A range of plausible values keeps the card's reason; a range
# that says who the study is about, such as the cards' age from 20, records the study
# population's, which the teaching content states and the app's card records too. Both reasons
# are written here by hand, from the ruling.
PLAUSIBLE = "implausible values (your own range)"
POPULATION = "outside the study population (your own range)"


def test_a_range_of_who_the_study_is_about_records_the_study_populations_reason():
    q = exclusions()
    sets = ["column=age", "low=20"]
    assert opts.resolve(q, ["eligibility"], sets) == {"kind": "set_exclusions", "rules": [
        {**RANGE_20, "reason": POPULATION}]}
    # silent: a range of plausible values keeps its own reason, alone or beside the other
    assert opts.resolve(q, ["custom"], sets)["rules"] == [{**RANGE_20, "reason": PLAUSIBLE}]
    assert [r["reason"] for r in opts.resolve(q, ["custom", "eligibility"], [
        "1.column=kcal", "1.high=4000", "2.column=age", "2.low=20"])["rules"]] == [
        PLAUSIBLE, POPULATION]
    # it is set as the other range is: a bound and a column, or refused
    with pytest.raises(opts.Unanswerable) as caught:
        opts.resolve(q, ["eligibility"], ["column=age"])
    assert "low" in str(caught.value)


def test_the_study_populations_reason_is_the_teaching_contents_and_the_cards(monkeypatch):
    """The driver types no reason of its own for it: it records the one the teaching content
    states, read when the option is composed (changed there, the driver follows), and the app's
    card records the same words (ChoiceQuestions.tsx's ``ELIGIBILITY_REASON``, read from its
    source)."""
    from research.swarm import driver
    from turbotab.core.teaching import content

    assert content.ELIGIBILITY_RANGE_REASON == POPULATION
    card = driver.REPO / "turbotab/frontend/src/components/record/ask/ChoiceQuestions.tsx"
    assert re.findall(r'export const ELIGIBILITY_REASON = "([^"]*)";', card.read_text()) == [
        POPULATION]
    monkeypatch.setattr(content, "ELIGIBILITY_RANGE_REASON", "outside the cohort (a test)")
    reasons = [opts.resolve(exclusions(), [option], ["column=age", "low=20"])["rules"][0]["reason"]
               for option in ("eligibility", "custom")]
    assert reasons == ["outside the cohort (a test)", PLAUSIBLE]


def _excluded_boxes(svg: str) -> list[tuple[str, int]]:
    """The participant-flow figure's side boxes as it prints them: each box's text from its
    "Excluded:", its wrapped lines joined by a space, and the count beneath it."""
    import html

    lines = [html.unescape(t) for t in re.findall(r"<text[^>]*>([^<]*)</text>", svg)]
    boxes, i = [], 0
    while i < len(lines):
        if lines[i].startswith("Excluded: "):
            j = next(k for k in range(i, len(lines)) if lines[k].startswith("n = "))
            boxes.append((" ".join(lines[i:j]), int(lines[j][4:].replace(",", ""))))
            i = j
        i += 1
    return boxes


def test_the_flow_and_its_figure_say_why_each_range_left_its_rows():
    """Both ways, on a table built here: the driver's range from 20 on who the study is about,
    then its plausible range on energy, through the engine's participant flow and the figure the
    export draws from it. Each step carries its own rule's reason, and each box prints it beside
    the rows pandas counts that rule removing."""
    import numpy as np
    import pandas as pd

    from turbotab.core.export import figures
    from turbotab.core.stages import rows

    rng = np.random.default_rng(20)
    n = 500
    frame = pd.DataFrame({"age": rng.integers(12, 86, n),
                          "kcal": rng.normal(2000, 700, n).round(),
                          "glucose": rng.normal(100, 12, n).round(1)})
    rules = opts.resolve(exclusions(), ["eligibility", "custom"], [
        "1.column=age", "1.low=20", "2.column=kcal", "2.low=800", "2.high=4000"])["rules"]
    adult = frame["age"] >= 20
    young = int((~adult).sum())
    implausible = int((adult & ~frame["kcal"].between(800, 4000)).sum())
    assert young > 0 and implausible > 0
    steps, _ = rows.cohort_flow(frame, target="glucose", rules=rules, missing=None,
                                predictor_columns=[])
    by_key = {s["key"]: s for s in steps}
    assert [(by_key[f"exclusion:{i}"]["reason"], by_key[f"exclusion:{i}"]["dropped"])
            for i in (0, 1)] == [(POPULATION, young), (PLAUSIBLE, implausible)]
    svg, _ = figures.participant_flow({"steps": steps}, purpose="prediction", n_train=None,
                                      n_holdout=None)
    assert _excluded_boxes(svg) == [(f"Excluded: {POPULATION}", young),
                                    (f"Excluded: {PLAUSIBLE}", implausible)]


# Levels of one's own as the app's card records them (ChoiceQuestions.tsx's ``levelRule``, read by
# hand): the column; the levels excluded and the codes that record a refused or a don't-know
# answer, each a list as typed, by commas, each trimmed and a blank dropped; the card's reason; and
# a blank, refused or don't-know answer left out as not confirmed eligible (the server's default).
LEVEL_LISTS = ("exclude", "refused", "dont_know")
OWN_LEVELS = {"kind": "levels", "column": "sex", "exclude": ["2"], "refused": [], "dont_know": [],
              "reason": "not in the study population (your own levels)", "missing": "exclude"}


def test_levels_of_ones_own_record_the_rule_the_card_records():
    q = exclusions()
    assert opts.resolve(q, ["levels"], ["level_column=sex", "exclude=2"]) == {
        "kind": "set_exclusions", "rules": [OWN_LEVELS]}
    # several levels by commas, as typed: the server compares them in one spelling (``_level_key``)
    assert opts.resolve(q, ["levels"], ["level_column=age", "exclude= 1, 3,,", "refused=7",
                                        "dont_know=9,99"])["rules"] == [
        {**OWN_LEVELS, "column": "age", "exclude": ["1", "3"], "refused": ["7"],
         "dont_know": ["9", "99"]}]
    # a list set twice keeps the last, as a field does
    assert opts.resolve(q, ["levels"], ["level_column=sex", "exclude=1", "exclude=2"]) == {
        "kind": "set_exclusions", "rules": [OWN_LEVELS]}
    assert q["options"][-1]["decision"] is None  # the offer itself unchanged


@pytest.mark.parametrize("sets,word", [
    ([], "--set"),  # nothing set: it records nothing by itself
    (["exclude=1"], "level_column"),  # no column
    (["level_column=sex"], "exclude"),  # no level excluded
    (["level_column=sex", "refused=7"], "exclude"),
    (["level_column=sex", "exclude=, ,"], "exclude"),  # blanks only
    (["level_column=glucose", "exclude=1"], "level_column"),  # the outcome
    (["level_column=__row_id", "exclude=1"], "level_column"),
    (["level_column=height", "exclude=1"], "level_column"),  # not a column
    (["level_column=sex", "exclude=1", "refused=1"], "`1`"),  # one level, two meanings
    (["level_column=sex", "exclude=2,2.0"], "`2`"),  # one level twice, as the server compares
    (["level_column=sex", "exclude=1", "dont_know=7", "refused=7.0"], "`7`"),
    (["level_column=sex", "exclude=1", "low=1"], "low"),  # the range's bound
])
def test_levels_without_a_column_or_a_level_or_with_a_level_named_twice_are_refused(sets, word):
    """The card records nothing while its rule excludes no level or names one level twice
    (``levelRule`` returns null; ``LevelRule`` refuses it too), nor without a column it lists."""
    with pytest.raises(opts.Unanswerable) as caught:
        opts.resolve(exclusions(), ["levels"], sets)
    assert word in str(caught.value), (sets, str(caught.value))


def test_what_the_levels_option_records_the_engine_accepts():
    """Silent: each level named once, the decision ``resolve`` builds parses as the engine's
    eligibility rule by level, its lists as given."""
    from turbotab.core.decisions import LevelRule, parse_decision

    decision = parse_decision(opts.resolve(exclusions(), ["levels"], [
        "level_column=sex", "exclude=1,3", "refused=7", "dont_know=9"]))
    rule = decision.rules[0]
    assert isinstance(rule, LevelRule) and rule.column == "sex" and rule.missing == "exclude"
    assert (rule.exclude, rule.refused, rule.dont_know) == (["1", "3"], ["7"], ["9"])


# ── several rules in one answer (the final fix round's A1) ───────────────────
# A card's population is often several rules: Bang 2009's is an age range and four level rules
# (research/datasets/bang2009/card.json), each with its own handling of a row it cannot confirm.
# One answer records them all, in the order named; a setting goes to the one option named that
# takes it, or, where several do, to the one its place names (``--set 2.level_column=DIQ010``).
# The expected rules are written here by hand from the constructed card above.

RANGE_20 = {"kind": "range", "column": "age", "low": 20.0, "high": None, "by": None,
            "reason": "implausible values (your own range)", "missing": "exclude"}


def engine_handlings() -> dict[str, tuple[str, ...]]:
    """What each of the engine's rule models takes for a row it cannot confirm (its ``missing``
    field's literal values), read off the models themselves."""
    from typing import get_args

    from turbotab.core.decisions import ExclusionRule, LevelRule

    return {m.__name__: get_args(m.model_fields["missing"].annotation)
            for m in (ExclusionRule, LevelRule)}


def test_one_answer_records_several_rules_each_set_by_its_place():
    q = exclusions()
    assert q["multi"] == "rules"
    got = opts.resolve(q, ["custom", "levels", "levels"], [
        "1.column=age", "1.low=20", "2.level_column=sex", "2.exclude=2", "2.missing=keep",
        "3.level_column=smoker", "3.exclude=1", "3.refused=7", "3.dont_know=9"])
    assert got == {"kind": "set_exclusions", "rules": [
        RANGE_20, {**OWN_LEVELS, "missing": "keep"},
        {**OWN_LEVELS, "column": "smoker", "exclude": ["1"], "refused": ["7"],
         "dont_know": ["9"]}]}
    # a setting only one option named takes needs no place; an offered screen adds its own rule
    assert opts.resolve(q, ["sex_neutral_500_3500", "levels"], [
        "level_column=sex", "exclude=2"]) == {"kind": "set_exclusions",
                                              "rules": [SCREEN, OWN_LEVELS]}
    assert opts.resolve(q, ["custom", "levels"], ["column=age", "low=20", "level_column=sex",
                                                  "exclude=2", "2.missing=keep"])["rules"] == [
        RANGE_20, {**OWN_LEVELS, "missing": "keep"}]
    # one option: its place may be named, and a range's own handling is set as a level rule's
    assert opts.resolve(q, ["custom"], ["1.column=age", "low=20", "missing=keep"])["rules"] == [
        {**RANGE_20, "missing": "keep"}]
    assert q["options"][-1]["decision"] is None and q["options"][-2]["decision"] is None


def test_what_several_rules_record_the_engine_reads_rule_by_rule():
    """Silent: the decision parses as the engine's own answer, each rule of the kind named and
    with the handling set for it."""
    from turbotab.core.decisions import ExclusionRule, LevelRule, parse_decision

    decision = parse_decision(opts.resolve(exclusions(), ["custom", "levels", "levels"], [
        "1.column=age", "1.low=20", "2.level_column=sex", "2.exclude=2", "2.missing=keep",
        "3.level_column=smoker", "3.exclude=1", "3.refused=7", "3.dont_know=9"]))
    assert [type(r) for r in decision.rules] == [ExclusionRule, LevelRule, LevelRule]
    assert [r.missing for r in decision.rules] == ["exclude", "keep", "exclude"]
    assert (decision.rules[2].refused, decision.rules[2].dont_know) == (["7"], ["9"])


def test_a_rules_handling_of_what_it_cannot_confirm_is_the_engines_choice():
    """The field ``missing`` offers what the engine's rule models take, read off the models, and
    starts where the card records it (a row the rule cannot confirm left out)."""
    field = next(f for f in exclusions()["fields"] if f["name"] == "missing")
    handlings = engine_handlings()
    assert handlings["ExclusionRule"] == handlings["LevelRule"]
    assert [c["value"] for c in field["choices"]] == list(handlings["LevelRule"])
    assert (field["initial"], field["applies_to"]) == ("exclude", ["custom", "eligibility",
                                                                    "levels"])


@pytest.mark.parametrize("ids,sets,words", [
    # a setting two options named take, with no place: each place is named
    (["custom", "levels"], ["column=age", "low=20", "level_column=sex", "exclude=2",
                            "missing=keep"], ["1.missing", "2.missing"]),
    (["levels", "levels"], ["level_column=sex", "exclude=2"], ["1.level_column"]),
    # the second rule unset, or half set: its own fields by its place
    (["levels", "levels"], ["1.level_column=sex", "1.exclude=2"], ["Option 2 (levels)",
                                                                   "2.level_column"]),
    (["levels", "levels"], ["1.level_column=sex", "1.exclude=2", "2.level_column=age"],
     ["Option 2 (levels)", "2.exclude"]),
    (["custom", "custom"], ["1.column=age", "1.low=20", "2.column=kcal"],
     ["Option 2 (custom)", "2.low"]),
    # a place past the options named, or one whose option does not take the setting
    (["levels"], ["2.level_column=sex", "exclude=2"], ["2", "1 is named"]),
    (["custom", "levels"], ["1.level_column=sex", "1.exclude=2"],
     ["does not apply to option custom"]),
    # keeping every row records no rule beside another
    (["none", "levels"], ["level_column=sex", "exclude=2"], ["none", "alone"]),
    # a handling the engine does not take
    (["levels"], ["level_column=sex", "exclude=2", "missing=maybe"], ["exclude", "keep"]),
    (["levels", "levels"], ["1.level_column=sex", "1.exclude=2", "2.level_column=age",
                            "2.exclude=1", "2.refused=1"], ["Option 2 (levels)", "`1`"]),
])
def test_several_rules_left_unset_or_set_out_of_place_are_refused(ids, sets, words):
    with pytest.raises(opts.Unanswerable) as caught:
        opts.resolve(exclusions(), ids, sets)
    assert all(w in str(caught.value) for w in words), str(caught.value)


def test_a_way_forward_beside_other_options_is_refused_for_why_it_is_refused():
    """A question that takes several options refuses a way forward named beside them, naming
    the way and saying that it is answered alone and that only the question's own options are
    named together (the driver fix package verifier's P1); a question that takes one option keeps saying
    so. Silent: the way alone, and the question's own options together, are answered."""
    keep = {"kind": "set_exclusions", "rules": []}
    ways = opts.exits([{"label": "Keep every row instead", "decision": keep}])
    way = ways[0]["id"]
    q = exclusions()
    for ids in ([way, "levels"], ["levels", way]):
        with pytest.raises(opts.Unanswerable) as caught:
            opts.resolve(q, ids, ["level_column=sex", "exclude=2"], ways)
        assert str(caught.value) == (f"Option {way} is a way forward: it is answered alone, and "
                                     f"only this question's own options are named together.")
    other = opts.exits([{"label": "Wait", "decision": {"kind": "set_lens", "lenses": []}}])
    with pytest.raises(opts.Unanswerable) as caught:
        opts.resolve(q, [way, other[0]["id"], "levels"], ["level_column=sex", "exclude=2"],
                     [*ways, *other])
    assert str(caught.value).startswith(f"Options {way} and {other[0]['id']} are ways forward: "
                                        f"each is answered alone"), str(caught.value)
    assert opts.resolve(q, [way], [], ways) == keep
    assert opts.resolve(q, ["sex_neutral_500_3500", "levels"], ["level_column=sex", "exclude=2"],
                        ways)["rules"] == [SCREEN, OWN_LEVELS]
    one = compose("missing", {"proposals": {"labels": {"missing": LABELED(
        "missing", ["impute", "complete_case"])}}})
    for ids in (["impute", way], ["impute", "complete_case"]):
        with pytest.raises(opts.Unanswerable) as caught:
            opts.resolve(one, ids, [], ways)
        assert str(caught.value) == "This question takes one option at a time."


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


# ── Bang's population, answered through the driver and counted by the engine ─
# Bang et al. 2009 left out the people told they have diabetes (NHANES DIQ010 1), its refused (7)
# and don't-know (9) codes apart (SWARM_PLAN §5). The analyst records it as ``answer levels --set
# level_column=DIQ010 --set exclude=1 --set refused=7 --set dont_know=9``: parsed by the driver's
# own parser, composed and resolved by ``driver.chosen`` from what an in-process analyst-mode
# server serves, and posted by ``driver.post_decision``. The participant flow's counts are checked
# against pandas over the table written here, never against the engine.


class _InProcess:
    """The driver's ``Client`` on FastAPI's ``TestClient``: the same paths under ``/api``, each
    answer as the driver's ``Response``."""

    def __init__(self, client: Any):
        self.c = client

    def request(self, method: str, path: str, *, body: bytes | None = None,
                headers: dict[str, str] | None = None, query: dict[str, Any] | None = None) -> Any:
        from research.swarm import driver

        q = {k: v for k, v in (query or {}).items() if v is not None}
        r = self.c.request(method, "/api" + path, content=body, headers=headers or {}, params=q)
        return driver.Response(r.status_code, r.content, dict(r.headers))

    def get(self, path: str, **query: Any) -> Any:
        return self.request("GET", path, query=query)

    def post(self, path: str, data: Any = None, **query: Any) -> Any:
        body = None if data is None else json.dumps(data).encode("utf-8")
        return self.request("POST", path, body=body, query=query,
                            headers={"Content-Type": "application/json"} if body else {})


def _diabetes_table(path: Any) -> Any:
    """400 adults: DIQ010 1 yes, 2 no, 3 borderline, 7 refused, 9 don't know, or blank; glucose
    blank on a few rows, so the rule is counted among the rows with the outcome recorded."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(2009)
    n = 400
    codes = rng.choice([1, 2, 3, 7, 9, -1], n, p=[0.1, 0.7, 0.06, 0.05, 0.05, 0.04])
    frame = pd.DataFrame({
        "SEQN": np.arange(1, n + 1),
        "age": rng.integers(20, 80, n),
        "bmi": np.round(rng.normal(28, 5, n), 1),
        "DIQ010": pd.array([None if c == -1 else int(c) for c in codes], dtype="Int64"),
    })
    frame["LBXGLU"] = np.round(70 + 0.3 * frame["age"] + 0.8 * frame["bmi"]
                               + rng.normal(0, 8, n), 1)
    frame.loc[rng.choice(n, 12, replace=False), "LBXGLU"] = np.nan
    frame.to_csv(path, index=False)
    return frame


def test_bangs_population_is_answered_by_its_code_through_the_driver_and_counted_by_the_flow(
        tmp_path, monkeypatch):
    import functools

    from research.swarm import driver
    from turbotab.core import config
    from turbotab.core.tests.acceptance.server_drive import local_server
    from turbotab.server.tests.conftest import answer_settled, declare, prepare, wait_for

    frame = _diabetes_table(tmp_path / "bang_like.csv")
    monkeypatch.setattr(config, "Settings", functools.partial(config.Settings, analyst=True))
    monkeypatch.setenv("TURBOTAB_CLAIMS_STRICT", "1")
    with local_server(tmp_path / "home") as client:
        made = client.post("/api/projects", json={"path": str(tmp_path / "bang_like.csv")})
        assert made.status_code == 200, made.text
        pid = made.json()["id"]
        # What a user who knows the table answers when asked (as _diabetes_table writes it).
        declare(pid, {"code_or_count:age": "amount", "code_or_count:DIQ010": "code",
                      "code_or_count:SEQN": "code", "unit:age": "years"}, fixture="bang_like.csv")
        wait_for(client, pid, {"ingest": "fresh", "profile": "fresh"}, timeout=120)
        for body in ({"kind": "set_lens", "lenses": ["clinical"]},
                     {"kind": "set_target", "column": "LBXGLU"},
                     {"kind": "set_purpose", "purpose": "prediction"}):
            prepare(client, pid, body)
            assert answer_settled(client, pid, None, body).status_code == 200, body
        prepare(client, pid, {"kind": "set_exclusions", "rules": []})

        run_id = "run_01-" + "0" * 32
        args = driver.build_parser().parse_args([
            "answer", "--run", run_id, "levels", "--set", "level_column=DIQ010",
            "--set", "exclude=1", "--set", "refused=7", "--set", "dont_know=9",
            "--marker", "rows", "--reason", "Bang 2009 left out known diabetes."])
        run = driver.Run(tmp_path / run_id, {"project": pid})
        run.folder.mkdir()
        served = _InProcess(client)
        decision, q = driver.chosen(run, served, list(args.option), args.set)
        assert q["question"]["key"] == "exclusions"
        assert decision == {"kind": "set_exclusions", "rules": [{
            "kind": "levels", "column": "DIQ010", "exclude": ["1"], "refused": ["7"],
            "dont_know": ["9"], "reason": "not in the study population (your own levels)",
            "missing": "exclude"}]}
        posted = driver.post_decision(run, served, decision, args.reason)
        assert posted.status == 200, posted.body[:600]

        recorded = frame["LBXGLU"].notna()
        code = frame["DIQ010"]
        n = {"not_recorded": int((recorded & code.isna()).sum()),
             "refused": int((recorded & code.eq(7).fillna(False)).sum()),
             "dont_know": int((recorded & code.eq(9).fillna(False)).sum()),
             "at_level": int((recorded & code.eq(1).fillna(False)).sum())}
        assert min(n.values()) > 0 and int(recorded.sum()) < len(frame)
        unconfirmed = n["not_recorded"] + n["refused"] + n["dont_know"]
        assert posted.json()["decisions"][-1]["sentence"] == (
            f"`{n['at_level']}` rows with `DIQ010` `1` were excluded as not in the study "
            f"population (your own levels), and `{unconfirmed}` with `DIQ010` refused (`7`), don't "
            f"know (`9`) or not recorded were excluded as not confirmed eligible.")
        wait_for(client, pid, {"cohort": "fresh"}, timeout=180)
        steps = client.get(f"/api/projects/{pid}/stages/cohort").json()["artifact"]["steps"]
        dropped = {s["key"]: s["dropped"] for s in steps}
        assert dropped["outcome_measured"] == int((~recorded).sum())
        assert {k: dropped.get(f"exclusion:0:{k}") for k in ("not_recorded", "refused",
                                                               "dont_know")} == {
            k: n[k] for k in ("not_recorded", "refused", "dont_know")}
        assert dropped["exclusion:0"] == n["at_level"]
        kept = recorded & code.isin([2, 3]).fillna(False)
        assert next(s for s in steps if s["key"] == "exclusion:0")["n"] == int(kept.sum())


# ── a card's whole population in one answer, through the driver (A1) ─────────
# Each card's eligibility rules as the card states them (research/datasets/<id>/card.json, checked
# against its words below), with the handling of an unconfirmed row each rule is given here: the
# pregnancy rule keeps a blank (every man, asked of no man), as do the medication rules (asked only
# of some), while the diagnosis rule leaves out a blank, refused or don't-know answer. The age range
# is answered as who the study is about, and its rows leave for the study population's reason
# (V1). The rows the driver's one answer keeps are counted by the engine's participant flow and
# checked against a pandas filter of the same codes over the table written here, never against
# the engine.

def _level(column: str, refused: bool, missing: str) -> dict[str, Any]:
    return {"kind": "levels", "column": column, "exclude": ["1"],
            "refused": ["7"] if refused else [], "dont_know": ["9"] if refused else [],
            "missing": missing}


AGE_20 = {"kind": "range", "column": "RIDAGEYR", "low": 20.0, "missing": "exclude"}
POPULATIONS = {
    "bang2009": [AGE_20, _level("RIDEXPRG", False, "keep"), _level("DIQ010", True, "exclude"),
                 _level("DIQ050", True, "keep"), _level("DIQ070", True, "keep")],
    "nhanes_glucose_predict": [AGE_20, _level("DIQ010", True, "exclude"),
                               _level("DIQ050", True, "keep"), _level("DIQ070", True, "keep"),
                               _level("RIDEXPRG", False, "keep")],
}


def _population_table(path: Any, card: str) -> Any:
    """900 people aged 12 to 85, coded as NHANES codes them, with the skip patterns behind the
    blanks: RIDEXPRG asked of women aged 20 to 44 only, so blank for every man; DIQ070 asked only
    of a person told of diabetes or borderline (DIQ010 1 or 3); and, for the glucose card, DIQ050
    skipped in the 2017 cycle but for those (the card: its blanks all in 2017–March 2020) and asked
    of everyone before. The outcome is blank on about a third of the rows."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(2009 if card == "bang2009" else 2017)
    n = 900
    age = rng.integers(12, 86, n)
    male = rng.random(n) < 0.49
    diq010 = rng.choice([1, 2, 3, 7, 9], n, p=[0.10, 0.76, 0.06, 0.04, 0.04]).astype(float)
    diq010[rng.random(n) < 0.03] = np.nan
    told = np.isin(diq010, [1, 3])
    asked = ~male & (age >= 20) & (age <= 44)
    frame = pd.DataFrame({
        "SEQN": np.arange(1, n + 1), "RIDAGEYR": age, "RIAGENDR": np.where(male, 1, 2),
        "RIDEXPRG": np.where(asked, rng.choice([1, 2, 3], n, p=[0.12, 0.80, 0.08]), np.nan),
        "DIQ010": diq010,
        "DIQ050": rng.choice([1, 2, 7, 9], n, p=[0.05, 0.89, 0.03, 0.03]).astype(float),
        "DIQ070": np.where(told, rng.choice([1, 2, 7, 9], n, p=[0.5, 0.4, 0.05, 0.05]), np.nan),
        "BMXBMI": np.round(rng.normal(28, 5, n), 1)})
    if card == "nhanes_glucose_predict":
        frame["cycle_begin_year"] = rng.choice([2011, 2013, 2015, 2017], n)
        frame.loc[(frame["cycle_begin_year"] == 2017) & ~told, "DIQ050"] = np.nan
    else:
        frame.loc[rng.random(n) < 0.08, "DIQ050"] = np.nan
    frame["LBXGLU"] = np.round(70 + 0.3 * age + 0.8 * frame["BMXBMI"] + rng.normal(0, 8, n), 1)
    frame.loc[rng.random(n) < 0.33, "LBXGLU"] = np.nan
    for c in ("RIDEXPRG", "DIQ010", "DIQ050", "DIQ070"):
        frame[c] = frame[c].astype("Int64")
    frame.to_csv(path, index=False)
    return frame


def _passes(frame: Any, rule: dict[str, Any]) -> Any:
    """The rows ``rule`` keeps, by pandas over the codes: a range keeps a value at or above its
    low; a level rule keeps every level but those excluded; a blank (for a level rule also a
    refused or a don't-know code) stays only where the rule keeps what it cannot confirm."""
    col = frame[rule["column"]]
    blank = col.isna()
    if rule["kind"] == "range":
        inside = (col >= rule["low"]).fillna(False).astype(bool)
        return inside | blank if rule["missing"] == "keep" else inside
    codes = lambda levels: col.isin([int(v) for v in levels]).fillna(False).astype(bool)  # noqa: E731
    excluded = codes(rule["exclude"])
    if rule["missing"] == "keep":
        return ~excluded
    return ~excluded & ~blank & ~codes(rule["refused"]) & ~codes(rule["dont_know"])


def _answer_words(rules: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """The driver's one answer for ``rules``: an option per rule, each setting by its place."""
    ids, sets = [], []
    for i, rule in enumerate(rules, 1):
        if rule["kind"] == "range":  # the cards' age: who the study is about (V1)
            ids.append("eligibility")
            sets += [f"{i}.column={rule['column']}", f"{i}.low={rule['low']:g}"]
        else:
            ids.append("levels")
            sets += [f"{i}.level_column={rule['column']}",
                     *(f"{i}.{k}={','.join(rule[k])}" for k in LEVEL_LISTS if rule[k])]
        if rule["missing"] != "exclude":
            sets.append(f"{i}.missing={rule['missing']}")
    return ids, sets


def _stated_by_the_card(card: dict[str, Any], rule: dict[str, Any]) -> bool:
    """Whether the card's own eligibility sentence for the rule's column states its codes."""
    said = next(s for s in card["eligibility"] if f"rule on {rule['column']} " in s)
    if rule["kind"] == "range":
        return f"a range rule on {rule['column']} from {rule['low']:g}" in said
    named = (f"naming {rule['refused'][0]} as its refused code and {rule['dont_know'][0]} as its "
             f"don't-know code" if rule["refused"] else "which has no refused or don't-know code")
    return f"a level rule on {rule['column']} excluding level {rule['exclude'][0]} " in said \
        and named in said


@pytest.mark.parametrize("card_id", list(POPULATIONS))
def test_a_cards_population_is_one_answer_through_the_driver_and_keeps_the_rows_pandas_keeps(
        card_id, tmp_path, monkeypatch):
    import functools

    from research.swarm import driver
    from turbotab.core import config
    from turbotab.core.export import figures
    from turbotab.core.tests.acceptance.server_drive import local_server
    from turbotab.server.tests.conftest import answer_settled, declare, prepare, wait_for

    card = json.loads((driver.REPO / "research" / "datasets" / card_id / "card.json").read_text())
    rules = POPULATIONS[card_id]
    assert all(_stated_by_the_card(card, r) for r in rules)
    frame = _population_table(tmp_path / "population.csv", card_id)
    # Pandas over the codes: the rows recorded with the outcome, then each rule in its order.
    kept = frame["LBXGLU"].notna()
    after = []
    for rule in rules:
        kept = kept & _passes(frame, rule)
        after.append(int(kept.sum()))
    male = frame["RIAGENDR"].eq(1)
    pregnancy = next(i for i, r in enumerate(rules) if r["column"] == "RIDEXPRG")
    others = frame["LBXGLU"].notna()
    for i, rule in enumerate(rules):
        if i != pregnancy:
            others = others & _passes(frame, rule)
    # Construction: every man is blank on RIDEXPRG, so the rule's keep is what keeps them all.
    assert int((others & male).sum()) == int((kept & male).sum()) > 0
    assert int((kept & male & frame["RIDEXPRG"].notna()).sum()) == 0
    if card_id == "nhanes_glucose_predict":  # DIQ050's skip pattern: most of 2017 is blank
        later = frame["cycle_begin_year"].eq(2017)
        blank = frame["DIQ050"].isna()
        assert int((kept & later).sum()) > 0
        assert int((kept & later & blank).sum()) > int((kept & later & ~blank).sum())

    monkeypatch.setattr(config, "Settings", functools.partial(config.Settings, analyst=True))
    monkeypatch.setenv("TURBOTAB_CLAIMS_STRICT", "1")
    with local_server(tmp_path / "home") as client:
        made = client.post("/api/projects", json={"path": str(tmp_path / "population.csv")})
        assert made.status_code == 200, made.text
        pid = made.json()["id"]
        declare(pid, {**{f"code_or_count:{c}": "code" for c in (
            "SEQN", "RIAGENDR", "RIDEXPRG", "DIQ010", "DIQ050", "DIQ070")},
            "code_or_count:RIDAGEYR": "amount", "unit:RIDAGEYR": "years",
            "code_or_count:cycle_begin_year": "amount"}, fixture="population.csv")
        wait_for(client, pid, {"ingest": "fresh", "profile": "fresh"}, timeout=120)
        for body in ({"kind": "set_lens", "lenses": ["clinical"]},
                     {"kind": "set_target", "column": "LBXGLU"},
                     {"kind": "set_purpose", "purpose": "prediction"}):
            prepare(client, pid, body)
            assert answer_settled(client, pid, None, body).status_code == 200, body
        prepare(client, pid, {"kind": "set_exclusions", "rules": []})

        ids, sets = _answer_words(rules)
        words = [a for s in sets for a in ("--set", s)]
        run_id = "run_01-" + "0" * 32
        run = driver.Run(tmp_path / run_id, {"project": pid})
        run.folder.mkdir()
        served = _InProcess(client)
        monkeypatch.setattr(driver.Run, "load", classmethod(lambda cls, rid: run))
        monkeypatch.setattr(driver.Run, "client", lambda self: served)
        # The analyst previews the whole population first, then records it, in one command each.
        looked = driver.build_parser().parse_args([
            "preview", "--run", run_id, *ids, *words, "--reason", "The card's population."])
        code, previewed, _ = driver.cmd_preview(looked)
        assert code == 0, previewed
        args = driver.build_parser().parse_args([
            "answer", "--run", run_id, *ids, *words, "--marker", "rows",
            "--reason", "The card's population."])
        decision, q = driver.chosen(run, served, list(args.option), args.set)
        assert q["question"]["key"] == "exclusions" and previewed["previewed"] == decision
        reasons = {"range": POPULATION, "levels": "not in the study population (your own levels)"}
        assert decision == {"kind": "set_exclusions", "rules": [
            {**({"by": None, "high": None} if r["kind"] == "range" else {}), **r,
             "reason": reasons[r["kind"]]} for r in rules]}
        posted = driver.post_decision(run, served, decision, args.reason)
        assert posted.status == 200, posted.body[:600]

        wait_for(client, pid, {"cohort": "fresh"}, timeout=180)
        steps = {s["key"]: s for s in
                 client.get(f"/api/projects/{pid}/stages/cohort").json()["artifact"]["steps"]}
        assert steps["outcome_measured"]["n"] == int(frame["LBXGLU"].notna().sum())
        assert [steps[f"exclusion:{i}"]["n"] for i in range(len(rules))] == after
        # The men: blank on RIDEXPRG, kept on a line of the rule's own that says so.
        before = frame["LBXGLU"].notna()
        for rule in rules[:pregnancy]:
            before = before & _passes(frame, rule)
        line = steps[f"exclusion:{pregnancy}:not_recorded"]
        blanks = int((before & frame["RIDEXPRG"].isna()).sum())
        assert line["dropped"] == 0 and line["label"].endswith(f": `{blanks:,}` kept")
        # The age rule's line, and the box the exported figure draws for it: the study
        # population's reason beside the rows pandas counts it removing, never implausible values.
        age = next(i for i, r in enumerate(rules) if r["kind"] == "range")
        before = frame["LBXGLU"].notna()
        for rule in rules[:age]:
            before = before & _passes(frame, rule)
        young = int((before & ~_passes(frame, rules[age])).sum())
        assert young > 0
        assert (steps[f"exclusion:{age}"]["reason"], steps[f"exclusion:{age}"]["dropped"]) == (
            POPULATION, young)
        cohort = client.get(f"/api/projects/{pid}/stages/cohort").json()["artifact"]
        svg, _ = figures.participant_flow(cohort, purpose="prediction", n_train=None,
                                          n_holdout=None)
        boxes = _excluded_boxes(svg)
        assert (f"Excluded: {POPULATION}", young) in boxes
        assert not any(PLAUSIBLE in text for text, _ in boxes)
