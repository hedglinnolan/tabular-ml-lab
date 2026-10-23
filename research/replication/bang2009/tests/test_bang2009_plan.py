"""The pre-registration's shape: every choice the notes leave open is in it, the sweep moves one
decision at a time, and the run's answers are the plan's, or the run stops: a refused answer or
eligibility, or a question offering no option the plan names (``prereg.json``, ``run.py``)."""
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


# ── a refused answer stops the run ───────────────────────────────────────────

# The refusal's way forward, which is not the plan's answer: the unweighted estimate that once
# stood in the six-year weight's place (run.py's module docstring).
INSTEAD = {"label": "Estimate for these participants instead",
           "decision": {"kind": "set_survey", "estimand": "sample"}}


def replicator(tmp_path, exits: list[dict] | None = None):
    """The run's analyst; ``exits``: the ways forward its plan names (``Spec.exits``), none as
    run.py's own spec names."""
    a = run.Replicator(run.D.Spec(name="t", dataset="bang2009", target=PREREG["outcome"]["column"],
                                  event=str(PREREG["outcome"]["event"]), purpose="inference",
                                  lenses=["clinical"], readings={}, exits=list(exits or [])),
                       tmp_path)
    a.settings, a.kept = run.settings("estimate", "default"), {}
    return a


def scripted(a, monkeypatch, refuse: bool) -> tuple[list[list[str]], str]:
    """``a`` on a scripted driver: every ask is the survey question, offering the plan's answer
    alone; that answer is refused with one way forward (``INSTEAD``) when ``refuse``, and every
    other answer is recorded. Returns the ids of each answer posted, and the plan's option id."""
    from research.swarm import options as opts

    want = a.settings["survey"]
    plan = {"kind": "set_survey", **{k: v for k, v in want.items() if k != "about"}}
    question = {"question": {"key": "survey", "status": "open"},
                "options": [{"id": "s-plan", "label": "The six-year weight", "decision": plan}]}
    refusal = {"refused": {"by": "server", "status": 422, "code": "survey_design_refused",
                           "message": "scripted", "exits": opts.exits([INSTEAD])}}
    posted: list[list[str]] = []

    def cmd(name, *args):
        if name == "ask":
            return 0, question
        assert name == "answer"
        ids = list(args[:args.index("--reason")])
        posted.append(ids)
        if refuse and ids == ["s-plan"]:
            return 1, refusal
        return 0, {"recorded": ids}

    monkeypatch.setattr(a, "cmd", cmd)
    return posted, "s-plan"


def test_a_refused_answer_stops_the_run_and_its_way_forward_is_never_posted(monkeypatch, tmp_path):
    from research.swarm import options as opts

    exit_id = opts.exits([INSTEAD])[0]["id"]
    # The premise: on this refusal the dry run's analyst posts its way forward where the plan names
    # it (``Spec.exits``), and stops there itself where the plan names none.
    def unguarded(exits):
        u = replicator(tmp_path, exits)
        u.answer = lambda ids, sets, why, marker=run.D.DEFAULT_MARKER: \
            run.D.Analyst.answer(u, ids, sets, why, marker)
        return u

    u = unguarded([INSTEAD["decision"]])
    posted, plan = scripted(u, monkeypatch, refuse=True)
    u.until(lambda q: bool(posted))
    assert posted == [[plan], [exit_id]]
    u = unguarded(None)
    posted, plan = scripted(u, monkeypatch, refuse=True)
    with pytest.raises(run.D.Refused, match="names none of its ways forward"):
        u.until(lambda q: bool(posted))
    assert posted == [[plan]]

    # The run's analyst stops at the refusal with its own stop, before the dry run's reads it, and
    # posts nothing after it: with run.py's spec, naming no way forward, and even with one naming
    # this one.
    for exits in (None, [INSTEAD["decision"]]):
        a = replicator(tmp_path, exits)
        posted, plan = scripted(a, monkeypatch, refuse=True)
        with pytest.raises(RuntimeError, match="an answer was refused") as stopped:
            a.until(lambda q: bool(posted))
        assert not isinstance(stopped.value, run.D.Refused)  # the dry run catches only a Refused
        assert posted == [[plan]]
        assert exit_id not in {i for ids in posted for i in ids}

    a = replicator(tmp_path)
    posted, plan = scripted(a, monkeypatch, refuse=True)
    with pytest.raises(RuntimeError) as direct:
        a.answer([plan], [], "the plan's survey answer")
    assert not isinstance(direct.value, run.D.Refused) and posted == [[plan]]


def test_an_answer_the_server_records_passes_through_unchanged(monkeypatch, tmp_path):
    a = replicator(tmp_path)
    posted, plan = scripted(a, monkeypatch, refuse=False)
    assert a.answer([plan], [], "the plan's survey answer") == {"recorded": [plan]}
    a = replicator(tmp_path)
    posted, plan = scripted(a, monkeypatch, refuse=False)
    a.until(lambda q: bool(posted))
    assert posted == [[plan]]
    assert a.kept["answers"] == [{"ids": [plan], "sets": [],
                                  "why": "survey answered from the plan's survey answer "
                                         f"({a.settings['survey_answer']})."}]


# ── the eligibility the run posts itself stops the run where it is refused ──


def eligibility(a, monkeypatch, status: int, body: dict) -> tuple[list[list[str]], list[dict]]:
    """``a`` on a scripted driver: every ask is the exclusions question, the levels preview is
    answered, and the one ``set_exclusions`` the run posts gets ``status`` and ``body``. Returns
    each preview's arguments and each decision posted."""
    from research.swarm import driver

    question = {"question": {"key": "exclusions", "status": "open"}, "options": []}
    previews: list[list[str]] = []
    posts: list[dict] = []

    class Scripted:
        def client(self):
            return None

    def cmd(name, *args):
        if name == "ask":
            return 0, question
        assert name == "preview", name  # the run's only other command on this path
        previews.append(list(args))
        return 0, {"previewed": {"rows_left_out": 1}, "refused": None}

    def post(run_, client, decision, note):
        posts.append(decision)
        return driver.Response(status, json.dumps(body).encode(), {})

    monkeypatch.setattr(a, "cmd", cmd)
    monkeypatch.setattr(driver.Run, "load", staticmethod(lambda run_id: Scripted()))
    monkeypatch.setattr(driver, "post_decision", post)
    return previews, posts


REFUSED_RULES = {"error": {"code": "invalid_rules", "message": "scripted"}}


def test_a_refused_eligibility_stops_the_run_and_keeps_nothing_as_posted(monkeypatch, tmp_path):
    a = replicator(tmp_path)
    previews, posts = eligibility(a, monkeypatch, 422, REFUSED_RULES)
    with pytest.raises(RuntimeError, match="the eligibility was refused") as stopped:
        a.answer([run.RULES_ID], [], "the paper's population")
    assert "invalid_rules" in str(stopped.value)
    assert posts == [run.the_rules(a.settings)] and len(previews) == 1
    assert "eligibility_posted" not in a.kept
    rules = str(len(PREREG["eligibility"]["rules"]))
    assert a.log[-1] == {"args": ["post_decision", "set_exclusions", rules], "exit": 1,
                         "refused": "invalid_rules"}

    # Asked by the Router, the refusal ends the walk at the first post: the dry run's analyst
    # catches only a Refused, so no other answer follows it.
    a = replicator(tmp_path)
    previews, posts = eligibility(a, monkeypatch, 422, REFUSED_RULES)
    with pytest.raises(RuntimeError, match="the eligibility was refused") as stopped:
        a.until(lambda q: False)
    assert not isinstance(stopped.value, run.D.Refused) and len(posts) == 1


def test_a_recorded_eligibility_is_kept_as_posted_and_its_body_returned(monkeypatch, tmp_path):
    a = replicator(tmp_path)
    previews, posts = eligibility(a, monkeypatch, 200, {"recorded": "set_exclusions"})
    assert a.answer([run.RULES_ID], [], "the paper's population") == {"recorded": "set_exclusions"}
    plan = PREREG["eligibility"]["rules"]
    assert posts == [a.kept["eligibility_posted"]]
    assert a.kept["eligibility_posted"] == {"kind": "set_exclusions", "rules": list(plan.values())}
    assert a.log[-1]["exit"] == 0 and a.log[-1]["refused"] is None
    # The preview is the driver's own level rule for DIQ010, with the plan's codes.
    told = plan["told_diabetes"]
    sets = previews[0][previews[0].index("levels") + 1:previews[0].index("--reason")]
    assert sets == ["--set", f"level_column={told['column']}",
                    "--set", f"exclude={','.join(told['exclude'])}",
                    "--set", f"refused={','.join(told['refused'])}",
                    "--set", f"dont_know={','.join(told['dont_know'])}"]
    assert a.kept["driver_level_preview"] == {"exit": 0, "previewed": {"rows_left_out": 1},
                                              "refused": None}


# ── each answer the plan makes is the plan's, or the run stops ──────────────

SURVEY, MISSING = PREREG["survey"]["population"], PREREG["missing"]["strategy"]
# The options each question offers, by key, the plan's answer among them: the first-ranked option is
# never the plan's, so the app's first-ranked answer is not mistaken for it; and each near miss
# differs from it in one field.
PLAN_IDS = {"survey": "s-plan", "missing": "m-plan", "split": "0", "event": "1.0"}
OFFERED = {
    "survey": [{"id": "s-sample", "decision": {"kind": "set_survey", "estimand": "sample"}},
               {"id": "s-two-year", "decision": {"kind": "set_survey", **{
                   k: v for k, v in SURVEY.items() if k not in ("about", "four_year_weight")}}},
               {"id": "s-plan", "decision": {"kind": "set_survey", **{
                   k: v for k, v in SURVEY.items() if k != "about"}}}],
    "missing": [{"id": "m-impute", "decision": {"kind": "set_missing", "strategy": "impute"}},
                {"id": "m-indicators", "decision": {"kind": "set_missing", "strategy": MISSING,
                                                    "indicators": True}},
                {"id": "m-category", "decision": {"kind": "set_missing", "strategy": MISSING,
                                                  "categorical": "missing_category"}},
                {"id": "m-plan", "decision": {"kind": "set_missing", "strategy": MISSING}}],
    "split": [{"id": "0.2", "decision": {"kind": "set_split", "holdout": 0.2}},
              {"id": "0", "decision": {"kind": "set_split", "holdout": 0.0}}],
    "event": [{"id": "0.0"}, {"id": "1.0"}],
}
LANE = {"survey": "estimate", "missing": "estimate", "split": "predict", "event": "estimate"}
# Each stop's words before the options it lists: what it looked for, and the plan's answer in it.
NAMED = {"survey": ("no survey option records the plan's", SURVEY["four_year_weight"]),
         "missing": ("no missing-values option records", MISSING),
         "split": ("no split option holds out", str(PREREG["lanes"]["predict"]["holdout"])),
         "event": ("no event level", str(PREREG["outcome"]["event"]))}


@pytest.mark.parametrize("key", sorted(PLAN_IDS))
def test_a_question_offering_the_plan_s_answer_is_answered_with_it(key, tmp_path):
    a = replicator(tmp_path)
    a.settings = run.settings(LANE[key], "default")
    ids, sets, _ = a.choose(key, {"options": OFFERED[key]})
    assert ids == [PLAN_IDS[key]] and sets == []


@pytest.mark.parametrize("key", sorted(PLAN_IDS))
def test_a_question_without_the_plan_s_answer_stops_the_run_naming_it(key, tmp_path):
    a = replicator(tmp_path)
    a.settings = run.settings(LANE[key], "default")
    without = [o for o in OFFERED[key] if o["id"] != PLAN_IDS[key]]
    with pytest.raises(RuntimeError) as stopped:
        a.choose(key, {"options": without})
    head = str(stopped.value).split(": [")[0]
    looked_for, plan = NAMED[key]
    assert head.startswith(looked_for) and plan in head[len(looked_for):]
