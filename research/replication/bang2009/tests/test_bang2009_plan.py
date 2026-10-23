"""The pre-registration's shape: every choice the notes leave open is in it, the sweep moves one
decision at a time, and the run's answers are the plan's (``prereg.json``, ``run.py``)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from research.replication.bang2009 import run

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[2]
PREREG = json.loads((HERE / "prereg.json").read_text("utf-8"))
NOTES = (REPO / PREREG["paper"]["notes"]).read_text("utf-8")


def section(text: str, heading: str) -> str:
    start = text.index(heading)
    end = text.find("\n## ", start + len(heading))
    return text[start:end if end > 0 else None]


def test_every_choice_the_notes_leave_open_is_decided_or_fixed():
    open_choices = re.findall(r"^(\d+)\. \*\*", section(NOTES, "## 6. "), flags=re.M)
    assert open_choices == [str(i) for i in range(1, len(open_choices) + 1)] and open_choices
    named = {m for item in [*PREREG["decisions"], *PREREG["fixed"]]
             for m in re.findall(r"§6 choice (\d+)", item["notes"])}
    assert named == set(open_choices)


def test_each_variant_moves_one_decision_and_every_other_option_is_swept():
    decisions = {d["id"]: d for d in PREREG["decisions"]}
    variants = PREREG["sweep"]["variants"]
    assert variants["default"] == {}
    swept = set()
    for name, changed in variants.items():
        if name == "default":
            continue
        assert len(changed) == 1, name
        (decision, option), = changed.items()
        assert option in decisions[decision]["options"] and option != decisions[decision]["default"]
        swept.add((decision, option))
    every_other = {(d, o) for d, spec in decisions.items() for o in spec["options"]
                   if o != spec["default"]}
    assert swept == every_other


def test_a_default_option_changes_nothing_and_each_variant_only_its_own_decision():
    base = run.settings("estimate", "default")
    assert base["rules"] == PREREG["eligibility"]["rules"]
    assert base["predictors"] == {k: p["column"] for k, p in PREREG["model"]["predictors"].items()}
    assert base["survey"]["estimand"] == "population" and base["exposure"] == "famhx"
    first = run.settings("estimate", "bp_first_reading")
    assert first["predictors"] == {**base["predictors"], "hypertension": "htn_first"}
    assert first["rules"] == base["rules"] and first["survey"] == base["survey"]
    fasting = run.settings("estimate", "fasting_any_glucose")
    assert set(base["rules"]) - set(fasting["rules"]) == {"fasting"}
    unknown = run.settings("estimate", "unknown_excluded")
    changed = {k for k in base["rules"] if base["rules"][k] != unknown["rules"][k]}
    assert changed == {"told_diabetes", "insulin"}
    assert {unknown["rules"][k]["missing"] for k in changed} == {"exclude"}
    family = run.settings("estimate", "family_parent_or_sibling")
    assert family["exposure"] == "famhx_close"  # the exposure follows its predictor's option
    assert run.settings("estimate", "unweighted")["survey"]["estimand"] == "sample"


def test_the_unweighted_variant_runs_under_inference_alone():
    names = run.variants()
    assert "estimate:unweighted" in names and "predict:unweighted" not in names
    assert {n.split(":")[1] for n in names if n.startswith("predict:")} == \
        set(PREREG["sweep"]["variants"]) - {"unweighted"}
    with pytest.raises(SystemExit):
        run.settings("predict", "unweighted")


def test_the_roles_put_the_six_predictors_in_the_model_and_leave_the_rest_out(monkeypatch):
    columns = ["SEQN", "SDDSRVYR", "SDMVSTRA", "SDMVPSU", "WTSAF2YR", "WTSAF4YR", "RIDAGEYR",
               "DIQ010", "diabetes_fpg", "age_band", "male", "famhx", "famhx_close", "htn",
               "htn_first", "obesity", "active"]
    monkeypatch.setattr(run, "table_columns", lambda: columns)
    e = run.roles_for(run.settings("estimate", "default"))
    assert {c for c, r in e.items() if r in ("exposure", "covariate")} == \
        {"age_band", "male", "famhx", "htn", "obesity", "active"}
    assert e["famhx"] == "exposure" and e["htn_first"] == e["famhx_close"] == "excluded"
    assert {c for c, r in e.items() if r == "design"} == \
        {"SDDSRVYR", "SDMVSTRA", "SDMVPSU", "WTSAF2YR", "WTSAF4YR"}
    assert "diabetes_fpg" not in e and e["SEQN"] == "identifier"
    p = run.roles_for(run.settings("predict", "bp_first_reading"))
    assert {c for c, r in p.items() if r == "exposure"} == \
        {"age_band", "male", "famhx", "htn_first", "obesity", "active"}
    assert p["htn"] == "excluded"


def test_the_eligibility_is_one_answer_of_every_rule_in_the_engine_s_shapes():
    from turbotab.core.decisions import SetExclusions

    for name in PREREG["sweep"]["variants"]:
        decision = run.the_rules(run.settings("estimate", name))
        parsed = SetExclusions.model_validate(decision)
        assert len(parsed.rules) == len(run.settings("estimate", name)["rules"])
