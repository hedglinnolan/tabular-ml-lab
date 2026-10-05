"""Each M1 validator refuses what the contract says, with exits the user can take instead."""
from __future__ import annotations

import pytest

from turbotab.core import decisions
from turbotab.core.decisions import ProjectState, Refusal, parse_decision, validate

COLUMNS = ["id", "age", "sex", "kcal", "protein_g", "fat_g", "carb_g", "sodium_mg", "note", "y", "cls",
           "site"]
INFO = {
    "id": {"dtype": "text", "n_unique": 100}, "age": {"dtype": "integer", "n_unique": 50},
    "sex": {"dtype": "categorical", "n_unique": 2}, "kcal": {"dtype": "numeric", "n_unique": 90},
    "protein_g": {"dtype": "numeric", "n_unique": 90}, "fat_g": {"dtype": "numeric", "n_unique": 90},
    "carb_g": {"dtype": "numeric", "n_unique": 90}, "sodium_mg": {"dtype": "numeric", "n_unique": 90},
    "note": {"dtype": "text", "n_unique": 80}, "y": {"dtype": "numeric", "n_unique": 70},
    "cls": {"dtype": "categorical", "n_unique": 3}, "site": {"dtype": "categorical", "n_unique": 40},
}
ROLES = {"id": "identifier", "age": "covariate", "sex": "covariate", "kcal": "energy",
         "protein_g": "exposure", "fat_g": "exposure", "carb_g": "exposure", "sodium_mg": "exposure",
         "note": "excluded", "site": "design"}


def ctx(**over):
    base = {"columns": COLUMNS, "column_info": INFO, "target": "y", "task": "regression",
            # The fixture's truth: `age` holds whole years, an amount (BLUEPRINT §14.3: whole
            # numbers settle nothing by their values).
            "state": ProjectState(lens=["dietary"], target="y", roles=ROLES,
                                  shape_confirmations={"code_or_count:age": "amount"})}
    return {**base, **over}


def refused(decision, context=None) -> Refusal:
    with pytest.raises(Refusal) as info:
        validate(decision, ctx() if context is None else context)
    for e in info.value.exits:  # every exit is a label and, when given, a decision that parses
        assert e["label"]
        if e["decision"] is not None:
            parse_decision(e["decision"])
    return info.value


def test_set_task_mismatch():
    r = refused({"kind": "set_task", "column": "y", "task": "binary"})
    assert r.code == "task_mismatch" and {e["decision"]["task"] for e in r.exits} == {"regression"}
    many = {**INFO, "y": {"dtype": "categorical", "n_unique": 25}}
    r = refused({"kind": "set_task", "column": "y", "task": "multiclass"}, ctx(column_info=many))
    assert r.code == "task_mismatch"
    validate({"kind": "set_task", "column": "y", "task": "regression"}, ctx())


def test_set_roles_refusals():
    assert refused({"kind": "set_roles", "roles": {"nope": "exposure", "age": "covariate"}}).code == "unknown_column"
    r = refused({"kind": "set_roles", "roles": {"y": "exposure", "age": "covariate"}})
    assert r.code == "target_has_role" and r.exits[0]["decision"]["roles"] == {"age": "covariate"}
    assert refused({"kind": "set_roles", "roles": {"id": "identifier", "note": "excluded"}}).code == "no_predictors"
    validate({"kind": "set_roles", "roles": ROLES}, ctx())


def test_set_exclusions_refusals():
    rule = {"column": "kcal", "low": 500, "high": 5000, "reason": "implausible"}
    assert refused({"kind": "set_exclusions", "rules": [{**rule, "column": "nope"}]}).code == "unknown_column"
    assert refused({"kind": "set_exclusions", "rules": [{**rule, "column": "sex"}]}).code == "not_numeric"
    r = refused({"kind": "set_exclusions", "rules": [{**rule, "low": 5000, "high": 500}]})
    assert r.code == "empty_range" and r.exits[0]["decision"]["rules"][0]["low"] == 500
    assert refused({"kind": "set_exclusions", "rules": [{**rule, "low": None, "high": None}]}).code == "no_bounds"
    validate({"kind": "set_exclusions", "rules": [rule]}, ctx())
    validate({"kind": "set_exclusions", "rules": []}, ctx())


def test_set_energy_adjustment_refusals():
    good = {"kind": "set_energy_adjustment", "method": "residual", "energy_column": "kcal",
            "nutrients": ["protein_g", "fat_g"]}
    validate(good, ctx())
    validate({"kind": "set_energy_adjustment", "method": "none"}, ctx())
    assert refused({**good, "energy_column": "age"}).code == "energy_role"
    assert refused({**good, "nutrients": []}).code == "nutrients_not_exposures"
    r = refused({**good, "nutrients": ["protein_g", "age"]})
    assert r.code == "nutrients_not_exposures" and r.exits[0]["decision"]["nutrients"] == ["protein_g"]
    r = refused({**good, "method": "partition", "nutrients": ["protein_g", "sodium_mg"]})
    assert r.code == "method_not_applicable"
    assert "partition" not in {e["decision"]["method"] for e in r.exits} and r.exits
    assert refused({**good, "strata": "site"}).code == "strata_not_categorical"
    validate({**good, "strata": "sex"}, ctx())
    no_roles = ctx(state=ProjectState(lens=["dietary"], target="y"))
    assert refused(good, no_roles).code == "roles_first"


def test_select_models_refusals():
    r = refused({"kind": "select_models", "models": ["linear", "quantum_forest"]})
    assert r.code == "unknown_model" and r.exits[0]["decision"]["models"] == ["linear"]
    validate({"kind": "select_models", "models": ["linear", "boosted_trees"]}, ctx())


def test_select_models_refuses_a_family_that_cannot_model_the_task(monkeypatch):
    monkeypatch.setattr(decisions, "model_families",
                        lambda: {"linear": {"regression", "binary"}, "boosted_trees": {"regression", "binary", "multiclass"}})
    r = refused({"kind": "select_models", "models": ["linear", "boosted_trees"]}, ctx(task="multiclass"))
    assert r.code == "model_cannot_fit_task" and r.exits[0]["decision"]["models"] == ["boosted_trees"]


def test_set_substitution_refusals():
    assert refused({"kind": "set_substitution", "donor": "fat_g", "recipient": "fat_g"}).code == "same_nutrient"
    r = refused({"kind": "set_substitution", "donor": "fat_g", "recipient": "sodium_mg"})
    assert r.code == "not_energy_bearing" and r.exits[0]["decision"]["donor"] == "fat_g"
    assert refused({"kind": "set_substitution", "donor": "fat_g", "recipient": "age"}).code == "not_energy_bearing"
    # BLUEPRINT §14.3 (amendment after the fifth gate): a name's ``_g`` never sets the kcal per
    # unit; with no values to read, the units are asked, and once recorded the swap stands.
    r = refused({"kind": "set_substitution", "donor": "fat_g", "recipient": "carb_g"})
    assert r.code == "reading_unsettled"
    from turbotab.core.decisions import ColumnUnitSpec

    grams = ctx()["state"].model_copy(update={"column_units": {
        c: ColumnUnitSpec(unit="g", days=None) for c in ("fat_g", "carb_g")}})
    validate({"kind": "set_substitution", "donor": "fat_g", "recipient": "carb_g"},
             ctx(state=grams))


def test_set_split_is_never_refused_for_small_n():
    validate({"kind": "set_split", "holdout": 0.4, "seed": 0, "folds": 10}, ctx(n_rows=5))


def test_a_sentence_is_authored_from_the_state_before_and_never_blocks(tmp_path):
    log = decisions.DecisionLog(tmp_path / "decisions.jsonl")
    log.append({"kind": "set_lens", "lenses": ["dietary"]})
    seen = []

    def sentence(decision, before):
        seen.append(before.target)
        return f"The outcome is `{decision.column}`."

    record = log.append({"kind": "set_target", "column": "y"}, sentence=sentence)
    assert record.sentence == "The outcome is `y`." and seen == [None]
    assert log.records()[-1].sentence == "The outcome is `y`."

    def broken(decision, before):
        raise RuntimeError("no voice today")

    record = log.append({"kind": "set_purpose", "purpose": "prediction"}, sentence=broken)
    assert record.sentence is None and log.state().purpose == "prediction"
