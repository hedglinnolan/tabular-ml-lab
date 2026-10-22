"""The open declaration lines' options (``declarations.py``; the sixth review's §0): every declaration
the quest log can open is composed, from what the engine defines for it, and answered like a
question. The truths are the engine's own definitions read here directly (the decision models'
fields, the method contracts' options for the purpose, the seal plan's served schemes), never the
composer's output; each guard both ways."""
from __future__ import annotations

import typing
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
    assert [c["value"] for c in got["sensitivity"]["choices"]] == ["true", "false"]
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
    (["sensitivity=yes"], "true or false"), (["where"], "name=value"), (["nope=1"], "nope")])
def test_a_field_set_outside_what_its_model_takes_is_refused(sets, word):
    got = D.compose("set_selection", view(), served())
    with pytest.raises(Unanswerable, match=word):
        D.resolve(got, ["set_selection:stability"], sets)


def test_a_field_set_inside_what_its_model_takes_is_recorded():
    got = D.compose("set_selection", view(), served())
    decision = D.resolve(got, ["set_selection:stability"],
                         ["threshold=0.7", "q=3", "pre_selected=no", "sensitivity=false"])
    assert (decision["threshold"], decision["q"], decision["pre_selected"]) == (0.7, 3, "no")
    assert isinstance(decision["q"], int)


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
