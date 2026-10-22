"""The open declaration lines' options (``declarations.py``; the sixth review's §0): every declaration
the quest log can open is composed, from what the engine defines for it, and answered like a
question. The truths are the engine's own definitions read here directly (the decision models'
fields, the method contracts' options for the purpose, the seal plan's served schemes), never the
composer's output; each guard both ways."""
from __future__ import annotations

import json
import typing
from pathlib import Path
from typing import Any

import pytest

from research.swarm import declarations as D
from research.swarm.options import Unanswerable
from turbotab.core import contracts, decisions

COLUMNS = ["y", "age", "x1", "x2", "plate", "site"]


def served(extra: dict[str, Any] | None = None, asked: list[str] | None = None):
    arts = {"ingest": {"columns": [{"name": c} for c in [*COLUMNS, "__row_id"]]}, **(extra or {})}

    def read(stage: str) -> Any:
        if asked is not None:
            asked.append(stage)
        return arts.get(stage)

    return read


def view(**state: Any) -> dict[str, Any]:
    return {"state": {"purpose": "prediction", "target": "y", **state}, "stages": {}}


def ids(composed: dict[str, Any]) -> list[str]:
    return [o["id"] for o in composed["options"]]


def test_every_declaration_the_quest_log_can_open_is_composed_but_those_that_need_a_file():
    from turbotab.core.quest import DECLARATIONS

    opened = [d.kind for d in DECLARATIONS if not d.only_recorded]
    assert D.kinds() == [k for k in opened if k not in D.NEEDS_A_FILE]
    assert set(D.NEEDS_A_FILE) == {"join_files", "import_codebook"}
    # The twelve the sixth review names, and the explanations: each reachable now.
    assert {"set_selection", "set_batch", "set_intended_use", "set_scales", "set_levers",
            "set_updating", "set_validation", "set_explain", "set_model_sequence",
            "set_measurement_error", "set_usual_intake", "set_multiplicity",
            "set_sensitivity"} == set(D.kinds())
    for kind in D.kinds():
        offered = ids(D.compose(kind, view(), served()))
        assert offered and len(set(offered)) == len(offered), kind  # each id names one option
    levers = ids(D.compose("set_levers", view(levers={"forms": "rule", "imbalance": "weights"}),
                           served()))
    assert "set_levers:none" in levers and "set_levers:forms:none" in levers


def test_the_selection_options_are_the_contracts_in_its_order_with_its_labels():
    got = D.compose("set_selection", view(), served())
    want = contracts.contract("variable_selection").options_for("prediction")
    assert ids(got) == [f"set_selection:{o['key']}" for o in want]
    for option, o in zip(got["options"], want):
        assert option["label"] == o["label"] and option["rung"] == o["rung"]
        assert option["labels"] == {"customary": o["customary"], "sound": o["sound"]}
        assert option["decision"]["method"] == o["key"]
    assert got["words"]["question"] == contracts.contract("variable_selection").question


def test_values_no_contract_labels_are_offered_unlabeled():
    """The batch contract's methods, then the model's one more (the column is not a batch)."""
    got = D.compose("set_batch", view(), served())
    batch = [o["key"] for o in contracts.contract("batch").options_for("prediction")]
    model = typing.get_args(decisions.BatchMethod)
    assert ids(got) == [f"set_batch:{k}" for k in batch] + [
        f"set_batch:{v}" for v in model if v not in batch]
    assert got["options"][-1]["label"] is None and got["options"][-1]["decision"] is not None


def test_each_option_records_a_decision_the_model_reads():
    """On a view whose answers give every field a model needs, each option's decision parses."""
    extra = {"proposals": {"model_sequence": {"decision": {"kind": "set_model_sequence",
                                                           "exposure": "x1", "model_1": ["age"]}}},
             "seal_plan": {"validation": {"options": [{"validation": "kfold", "label": "Once"},
                                                      {"validation": "repeated_kfold",
                                                       "label": "Repeated"}]}}}
    for kind in D.kinds():
        for option in D.compose(kind, view(purpose="inference"), served(extra))["options"]:
            if option["decision"] is not None:
                decisions.parse_decision(option["decision"])
            else:
                assert option.get("needs"), (kind, option["id"])


def test_a_field_takes_what_its_model_does():
    got = {f["name"]: f for f in D.compose("set_selection", view(), served())["fields"]}
    assert [c["value"] for c in got["pre_selected"]["choices"]] == list(typing.get_args(
        typing.get_args(decisions.SetSelection.model_fields["pre_selected"].annotation)[0]))
    assert got["threshold"]["bounds"] == {"gt": 0, "lt": 1}
    assert got["keep"]["bounds"] == {"ge": 1}
    validation = {f["name"]: f for f in D.compose("set_validation", view(), served())["fields"]}
    assert [c["value"] for c in validation["nested_cv"]["choices"]] == ["true", "false"]
    assert "method" not in got  # the options choose it
    intended = {f["name"]: f for f in D.compose("set_intended_use", view(), served())["fields"]}
    assert intended["subgroups"]["type"] == "columns"
    assert [c["value"] for c in intended["subgroups"]["choices"]] == COLUMNS  # never the row id


def test_each_field_starts_at_the_answer_recorded_now():
    levers = {"forms": "rule", "variance_filter": "none", "keep": None, "imbalance": "weights"}
    got = D.compose("set_levers", view(levers=levers), served())
    by_id = {o["id"]: o["decision"] for o in got["options"]}
    # The imbalance contract's "none" is the imbalance field's (every option of that contract is
    # its value), and it keeps the forms as recorded.
    assert by_id["set_levers:none"] == {"kind": "set_levers", "forms": "rule",
                                        "variance_filter": "none", "keep": None, "imbalance": "none"}
    assert by_id["set_levers:near_zero"]["imbalance"] == "weights"
    assert {f["name"]: f["initial"] for f in got["fields"]}["forms"] == "rule"
    # With nothing recorded, every field starts at the model's default.
    fresh = {f["name"]: f["initial"] for f in D.compose("set_levers", view(), served())["fields"]}
    assert fresh == {n: f.default for n, f in decisions.SetLevers.model_fields.items()
                     if n != "kind"}


def test_the_validation_schemes_are_the_seal_plans_and_without_one_the_models():
    plan = {"validation": {"options": [
        {"validation": "bootstrap", "label": "Bootstrap optimism correction", "measures": "m"},
        {"validation": "internal_external", "label": "By site", "cluster": "site"}],
        "reason": "why"}}
    got = D.compose("set_validation", view(), served({"seal_plan": plan}))
    assert ids(got) == ["set_validation:bootstrap", "set_validation:internal_external"]
    assert got["options"][0]["label"] == "Bootstrap optimism correction"
    assert got["options"][1]["decision"]["cluster"] == "site"
    assert got["data"] == {"reason": "why"}
    unplanned = D.compose("set_validation", view(), served())
    assert ids(unplanned) == [f"set_validation:{v}"
                              for v in typing.get_args(decisions.Validation)]


def test_the_sensitivity_analyses_are_the_stages_every_row_analysis_or_none():
    from turbotab.core.stages.sensitivity import EVERY_ROW

    got = D.compose("set_sensitivity", view(exclusions=[{"kind": "range"}]), served())
    assert [o["decision"] for o in got["options"]] == [
        {"kind": "set_sensitivity", "analyses": [{"label": EVERY_ROW, "rules": []}]},
        {"kind": "set_sensitivity", "analyses": []}]


def test_a_scale_of_ones_own_needs_its_fields_and_reads_as_the_model_reads_it():
    got = D.compose("set_scales", view(), served())
    assert ids(got) == ["set_scales:none", "set_scales:custom"]
    custom = got["options"][1]
    assert custom["decision"] is None
    assert custom["needs"] == [n for n, f in decisions.ScaleSpec.model_fields.items()
                               if f.is_required()]
    with pytest.raises(Unanswerable, match="takes name, items, low, high, kind"):
        D.resolve(got, ["set_scales:custom"], [])
    with pytest.raises(Unanswerable, match="at least 3"):
        D.resolve(got, ["set_scales:custom"], ["name=s", "items=age,x1", "low=0", "high=4",
                                                "kind=reflective"])
    decision = D.resolve(got, ["set_scales:custom"], ["name=s", "items=age,x1,x2", "low=0",
                                                       "high=4", "kind=reflective"])
    assert decision["scales"][0]["items"] == ["age", "x1", "x2"]
    decisions.parse_decision(decision)


def test_the_model_sequence_is_the_proposals_card_and_without_one_its_exposure_is_asked():
    card = {"decision": {"kind": "set_model_sequence", "exposure": "x1", "model_1": ["age"]},
            "label": "Model 1: age"}
    got = D.compose("set_model_sequence", view(purpose="inference"),
                    served({"proposals": {"model_sequence": card}}))
    assert ids(got) == ["set_model_sequence:proposed"]
    assert got["options"][0]["decision"] == card["decision"]
    bare = D.compose("set_model_sequence", view(purpose="inference"), served())
    assert ids(bare) == ["set_model_sequence:answer"] and bare["options"][0]["needs"] == ["exposure"]
    assert D.resolve(bare, ["set_model_sequence:answer"],
                     ["exposure=x1", "model_1=age"])["model_1"] == ["age"]


def test_a_batch_column_read_by_its_name_is_the_one_offered():
    got = D.compose("set_batch", view(lens=["metabolomics"]), served())
    column = next(f for f in got["fields"] if f["name"] == "column")
    assert [c["value"] for c in column["choices"]] == ["plate"]
    assert all(o["decision"]["column"] == "plate" for o in got["options"])


@pytest.mark.parametrize("sets,word", [
    (["threshold=0"], "threshold"), (["threshold=1"], "threshold"), (["keep=0"], "keep"),
    (["keep=1.5"], "whole number"), (["q=nan"], "q"), (["pre_selected=maybe"], "pre_selected"),
    (["where"], "name=value"), (["nope=1"], "nope")])
def test_a_field_set_outside_what_its_model_takes_is_refused(sets, word):
    got = D.compose("set_selection", view(), served())
    with pytest.raises(Unanswerable, match=word):
        D.resolve(got, ["set_selection:stability"], sets)


def test_a_true_or_false_field_takes_only_true_or_false():
    got = D.compose("set_validation", view(), served())
    with pytest.raises(Unanswerable, match="true or false"):
        D.resolve(got, ["set_validation:kfold"], ["nested_cv=yes"])
    assert D.resolve(got, ["set_validation:kfold"], ["nested_cv=true"])["nested_cv"] is True


def test_a_field_set_inside_what_its_model_takes_is_recorded():
    got = D.compose("set_selection", view(), served())
    decision = D.resolve(got, ["set_selection:stability"],
                         ["threshold=0.7", "q=3", "pre_selected=no"])
    assert (decision["threshold"], decision["q"], decision["pre_selected"]) == (0.7, 3, "no")
    assert isinstance(decision["q"], int)


# ── what the shell's cards hold fixed, and the attestations (the verifier's V1) ─────

CARDS = Path(__file__).resolve().parents[3] / (
    "turbotab/frontend/src/quest/cards/predict/payloads.fixture.json")


def held_by_the_cards() -> dict[str, dict[str, Any]]:
    """What the shell's Predict cards never ask and always post (the fixture's ``fixed``, which
    ``core/tests/test_w1_declaration_cards.py`` holds to the cards' payloads and to decisions.py's
    defaults), read here from the fixture itself."""
    return json.loads(CARDS.read_text("utf-8"))["fixed"]


def attestations(kind: str) -> list[str]:
    """The fields of ``kind``'s model that keep a blocked answer (decisions.py names each
    ``acknowledged``, or ``<what>_acknowledged``): true or false, false unless a refusal's way
    forward sets it."""
    model = D.models()[kind]
    return [n for n in model.model_fields if n == "acknowledged" or n.endswith("_acknowledged")]


def test_the_cards_hold_fixed_the_fields_the_verifier_named():
    """The construction the guard below rests on: the fixture holds where, the sensitivity
    analysis and the explanation as an effect fixed; every attestation in decisions.py is a true
    or false field false by default."""
    held = held_by_the_cards()
    assert held["set_selection"] == {"where": "in_fold", "sensitivity": False}
    assert held["set_explain"] == {"as_effect": False}
    assert attestations("set_multiplicity") == ["acknowledged"]
    assert "acknowledged" in attestations("set_validation")
    for kind, model in D.models().items():
        for name in attestations(kind):
            f = model.model_fields[name]
            assert f.annotation is bool and f.default is False, (kind, name)


@pytest.mark.parametrize("purpose", ["prediction", "inference"])
def test_no_field_the_cards_hold_fixed_and_no_attestation_is_offered(purpose):
    """``ask`` lists none of them, as a field or as an option, on any line, and every option
    records each one as the shell's cards post it (an attestation at its default, false)."""
    held = held_by_the_cards()
    extra = {"seal_plan": {"validation": {"options": [{"validation": "kfold", "label": "Once"}]}}}
    for kind in D.kinds():
        got = D.compose(kind, view(purpose=purpose, exclusions=[{"kind": "range"}]),
                        served(extra))
        hidden = {**held.get(kind, {}), **{n: False for n in attestations(kind)}}
        names = {f["name"] for f in got["fields"]}
        assert not names & set(hidden), (kind, names & set(hidden))
        for option in got["options"]:
            assert option["id"].split(":")[1] not in hidden, (kind, option["id"])
            recorded = option["decision"] or option.get("_base") or {}
            for name, value in hidden.items():
                if name in recorded:
                    assert recorded[name] == value, (kind, option["id"], name)


def test_the_fields_the_cards_do_send_stay_offered():
    held = held_by_the_cards()
    selection = {f["name"] for f in D.compose("set_selection", view(), served())["fields"]}
    assert selection == {"keep", "threshold", "q", "pre_selected"}
    explain = D.compose("set_explain", view(), served())
    assert {f["name"] for f in explain["fields"]} == {"exposures", "reseeds"}
    assert ids(explain) == ["set_explain:ale", "set_explain:partial_dependence"]
    validation = {f["name"] for f in D.compose("set_validation", view(), served())["fields"]}
    assert validation == {"folds", "repeats", "n_boot", "cluster", "nested_cv"}
    # Each is a field the model takes and the fixture does not hold.
    for kind, names in (("set_selection", selection), ("set_validation", validation)):
        assert names <= set(D.models()[kind].model_fields) - set(held[kind])


@pytest.mark.parametrize("kind,option,sets", [
    ("set_selection", "set_selection:stepwise", ["sensitivity=true"]),
    ("set_selection", "set_selection:elastic_net", ["sensitivity=true"]),
    ("set_selection", "set_selection:elastic_net", ["where=outside"]),
    ("set_validation", "set_validation:kfold", ["acknowledged=true"]),
    ("set_multiplicity", "set_multiplicity:none", ["acknowledged=true"]),
    ("set_explain", "set_explain:ale", ["as_effect=true"])])
def test_a_field_the_cards_hold_fixed_or_an_attestation_set_by_hand_is_refused(kind, option,
                                                                               sets):
    got = D.compose(kind, view(purpose="inference" if kind == "set_multiplicity" else
                               "prediction"), served())
    name = sets[0].split("=")[0]
    with pytest.raises(Unanswerable, match=f"No field '{name}'"):
        D.resolve(got, [option], sets)


def test_an_answer_recorded_with_a_held_field_set_is_posted_as_the_cards_post_it():
    """A recorded selection carrying the sensitivity analysis keeps its pre-selection answer but
    is posted with the cards' fixed values; a recorded attestation is never carried into another
    answer, which meets the engine's refusal and its way forward on its own."""
    got = D.compose("set_selection", view(selection={
        "method": "stepwise", "where": "outside", "sensitivity": True, "pre_selected": "no",
        "keep": None, "threshold": None, "q": None}), served())
    for option in got["options"]:
        decision = option["decision"]
        assert decision["sensitivity"] is False and decision["where"] == "in_fold"
        assert decision["pre_selected"] == "no"
    multiplicity = D.compose("set_multiplicity", view(
        purpose="inference", multiplicity={"method": "stated_count", "acknowledged": True}),
        served())
    assert all(o["decision"]["acknowledged"] is False for o in multiplicity["options"])


def test_one_option_at_a_time_and_only_one_offered():
    got = D.compose("set_updating", view(models=["linear"]), served())
    assert ids(got) == ["set_updating:none", "set_updating:shrinkage"]
    with pytest.raises(Unanswerable, match="one option"):
        D.resolve(got, ["set_updating:none", "set_updating:shrinkage"], [])
    with pytest.raises(Unanswerable, match="No option"):
        D.resolve(got, ["set_updating:sometimes"], [])
    assert D.resolve(got, ["set_updating:shrinkage"], []) == {"kind": "set_updating",
                                                              "method": "shrinkage"}


def test_an_option_id_names_its_declaration():
    declared = {"set_selection": {}, "set_batch": {}}
    assert D.owner("set_selection:vip", declared) == "set_selection"
    assert D.owner("set_selection", declared) is None  # no option of it
    assert D.owner("set_levers:rule", declared) is None  # a line not open
    assert D.owner("impute", declared) is None


def test_no_estimate_stage_is_read_to_compose_a_line():
    from turbotab.core.plan_lock import ESTIMATE_STAGES

    asked: list[str] = []
    for kind in D.kinds():
        D.compose(kind, view(), served(asked=asked))
    assert asked and not set(asked) & set(ESTIMATE_STAGES)


def test_a_field_the_server_fills_is_not_offered():
    got = D.compose("set_measurement_error", view(purpose="inference"), served())
    assert "adjustment" not in {f["name"] for f in got["fields"]}
    assert all("adjustment" not in (o["decision"] or {}) for o in got["options"])


# ── what an option needs before it records anything (the verifier's P9) ─────


def test_internal_external_validation_needs_the_cluster_the_seal_plan_says_it_needs():
    """The seal plan serves internal–external validation with ``needs_cluster`` (the engine's
    ``ValidationOption``): with no cluster named, the option records nothing by itself, says it
    needs ``cluster``, and takes one of the table's columns but the outcome, as the shell's
    validation card offers them."""
    from turbotab.core.models.validation import ValidationOption

    served_option = ValidationOption(validation="internal_external", label="By cluster",
                                     measures="m", cost="c", needs_cluster=True)
    plan = {"validation": {"options": [
        {"validation": "kfold", "label": "Once", "needs_cluster": False},
        served_option.model_dump(mode="json")]}}
    got = D.compose("set_validation", view(), served({"seal_plan": plan}))
    option = {o["id"]: o for o in got["options"]}
    assert option["set_validation:internal_external"]["decision"] is None
    assert option["set_validation:internal_external"]["needs"] == ["cluster"]
    assert option["set_validation:kfold"]["decision"] is not None
    assert not option["set_validation:kfold"].get("needs")
    cluster = {f["name"]: f for f in got["fields"]}["cluster"]
    assert [c["value"] for c in cluster["choices"]] == [c for c in COLUMNS if c != "y"]
    assert cluster["applies_to"] == ["set_validation:internal_external"]
    with pytest.raises(Unanswerable, match="cluster"):
        D.resolve(got, ["set_validation:internal_external"], [])
    decided = D.resolve(got, ["set_validation:internal_external"], ["cluster=site"])
    assert (decided["validation"], decided["cluster"]) == ("internal_external", "site")
    decisions.parse_decision(decided)
    with pytest.raises(Unanswerable, match="does not apply"):
        D.resolve(got, ["set_validation:kfold"], ["cluster=site"])


def test_internal_external_validation_with_the_cluster_the_plan_names_needs_nothing():
    plan = {"validation": {"options": [{"validation": "internal_external", "label": "By site",
                                        "needs_cluster": True, "cluster": "site"}]}}
    got = D.compose("set_validation", view(), served({"seal_plan": plan}))
    option = got["options"][0]
    assert option["decision"]["cluster"] == "site" and not option.get("needs")


def test_keeping_the_most_variable_predictors_needs_how_many_where_the_engine_refuses_it_without():
    """The engine refuses ``variance_filter: top`` with no ``keep`` (``keep_how_many``, its way
    forward setting ``keep``): the option says it needs ``keep``; given one, it records it."""
    with pytest.raises(decisions.Refusal) as refused:
        decisions.validate({"kind": "set_levers", "variance_filter": "top"}, None)
    assert refused.value.code == "keep_how_many"  # the engine's own rule, read here
    got = D.compose("set_levers", view(), served())
    option = {o["id"]: o for o in got["options"]}
    assert option["set_levers:top"]["decision"] is None
    assert option["set_levers:top"]["needs"] == ["keep"]
    assert option["set_levers:near_zero"]["decision"] is not None
    assert not option["set_levers:near_zero"].get("needs")
    with pytest.raises(Unanswerable, match="keep"):
        D.resolve(got, ["set_levers:top"], [])
    decided = D.resolve(got, ["set_levers:top"], ["keep=1000"])
    assert decided["variance_filter"] == "top" and decided["keep"] == 1000
    decisions.validate(decided, None)


def test_keeping_the_most_variable_with_a_keep_recorded_needs_nothing():
    levers = {"forms": "none", "variance_filter": "none", "keep": 500, "imbalance": "none"}
    got = D.compose("set_levers", view(levers=levers), served())
    option = {o["id"]: o for o in got["options"]}["set_levers:top"]
    assert option["decision"]["keep"] == 500 and not option.get("needs")


def test_with_no_seal_plan_internal_external_validation_still_needs_its_cluster():
    """The model itself does not read internal–external validation without a cluster
    (``decisions._validation_fields``): with no seal plan served, the option says it needs one."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="cluster"):
        decisions.parse_decision({"kind": "set_validation", "validation": "internal_external"})
    got = D.compose("set_validation", view(), served())
    option = {o["id"]: o for o in got["options"]}
    assert option["set_validation:internal_external"]["needs"] == ["cluster"]
    assert option["set_validation:internal_external"]["decision"] is None
    assert all(not o.get("needs") for i, o in option.items()
               if i != "set_validation:internal_external")
    decided = D.resolve(got, ["set_validation:internal_external"], ["cluster=site"])
    decisions.parse_decision(decided)
